// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Memory readings and per-scene peak aggregation.

use serde::Serialize;

/// A measurement or the concrete reason it is unavailable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Reading<T> {
    /// A value read from the named source.
    Measured(T),
    /// The source could not provide a value.
    Unavailable(String),
}

impl<T> Reading<T> {
    /// Records a measurement as unavailable without substituting a value.
    #[must_use]
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self::Unavailable(reason.into())
    }
}

/// Engine-accounted CPU and GPU bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct EngineBytes {
    /// Bytes reported for engine-owned CPU caches and resources.
    pub cpu_bytes: u64,
    /// Bytes reported for engine-owned GPU resources.
    pub gpu_bytes: u64,
}

/// Wgpu allocator totals.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AllocatorBytes {
    /// Bytes assigned to live allocations.
    pub allocated_bytes: u64,
    /// Bytes reserved in allocator blocks.
    pub reserved_bytes: u64,
    /// Number of live allocations.
    pub allocations: u64,
    /// Number of allocator blocks.
    pub blocks: u64,
}

/// Skia's API-specific budgeted resource bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SkiaBudget {
    /// Skia memory API that produced this value.
    pub api: &'static str,
    /// Budgeted resource bytes.
    pub bytes: u64,
}

/// A Vulkan memory heap's budget and current use.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VkHeap {
    /// Heap index reported by Vulkan.
    pub heap: u32,
    /// Whether the heap is device-local.
    pub device_local: bool,
    /// Bytes currently used on this heap.
    pub usage_bytes: u64,
    /// Vulkan's budget for this heap.
    pub budget_bytes: u64,
}

/// Memory sources available from one bench adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterMemory {
    /// Engine-accounted CPU and GPU bytes.
    pub engine: Reading<EngineBytes>,
    /// Wgpu allocator totals.
    pub wgpu_allocator: Reading<AllocatorBytes>,
    /// Skia budgeted GPU resource bytes.
    pub skia_budgeted: Reading<SkiaBudget>,
    /// Vulkan heap budget, when the adapter and platform expose it.
    pub vk_memory_budget: Reading<Vec<VkHeap>>,
}

/// Process-level memory measurements for the current platform.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "platform", rename_all = "snake_case")]
pub enum ProcessMemory {
    /// Linux resident set size and Vulkan heap budget.
    Linux {
        /// Resident set size in bytes.
        rss_bytes: Reading<u64>,
        /// Per-heap Vulkan use and budget.
        vk_memory_budget: Reading<Vec<VkHeap>>,
    },
    /// Apple task physical footprint.
    Apple {
        /// `TASK_VM_INFO.phys_footprint` in bytes.
        phys_footprint_bytes: Reading<u64>,
    },
    /// Android process PSS and graphics allocations.
    Android {
        /// Proportional set size in bytes.
        pss_bytes: Reading<u64>,
        /// `dumpsys meminfo` Graphics total in bytes.
        graphics_bytes: Reading<u64>,
        /// `dumpsys meminfo` GL mtrack PSS in bytes.
        gl_mtrack_bytes: Reading<u64>,
    },
    /// No process memory source is implemented for this operating system.
    Other {
        /// Rust's operating-system name.
        os: String,
    },
}

/// All memory sources sampled at one point in a scene.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MemorySnapshot {
    /// Engine-accounted CPU and GPU bytes.
    pub engine: Reading<EngineBytes>,
    /// Wgpu allocator totals.
    pub wgpu_allocator: Reading<AllocatorBytes>,
    /// Skia budgeted GPU resource bytes.
    pub skia_budgeted: Reading<SkiaBudget>,
    /// Process-level readings for the current platform.
    pub process: ProcessMemory,
}

/// Idle, steady-state and peak memory for one scene.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MemoryReport {
    /// Snapshot taken once after engine creation and before scene preparation.
    pub idle: MemorySnapshot,
    /// Snapshot after the last measured frame, or the readback submission for render.
    pub steady: MemorySnapshot,
    /// Field-wise maximum after prepare and every frame, including warmup,
    /// sampled outside the encode and submit timing intervals.
    pub peak: MemorySnapshot,
    /// Number of post-prepare and post-frame snapshots included in the peak window.
    pub samples: usize,
}

#[derive(Clone, Copy)]
pub(crate) enum SampleDetail {
    Full,
    Frame,
}

impl MemorySnapshot {
    pub(crate) fn capture(adapter: AdapterMemory, detail: SampleDetail) -> Self {
        let process = process_memory(&adapter.vk_memory_budget, detail);
        Self {
            engine: adapter.engine,
            wgpu_allocator: adapter.wgpu_allocator,
            skia_budgeted: adapter.skia_budgeted,
            process,
        }
    }

    fn peak(snapshots: &[Self]) -> Option<Self> {
        let first = snapshots.first()?;
        Some(Self {
            engine: merge_engine(snapshots.iter().map(|s| &s.engine)),
            wgpu_allocator: merge_allocator(snapshots.iter().map(|s| &s.wgpu_allocator)),
            skia_budgeted: merge_skia(snapshots.iter().map(|s| &s.skia_budgeted)),
            process: merge_process(snapshots.iter().map(|s| &s.process), &first.process),
        })
    }
}

impl MemoryReport {
    pub(crate) fn new(idle: MemorySnapshot, samples: &[MemorySnapshot]) -> Self {
        let steady = samples
            .last()
            .map_or_else(|| idle.clone(), |snapshot| (*snapshot).clone());
        let peak = MemorySnapshot::peak(samples).unwrap_or_else(|| steady.clone());
        Self {
            idle,
            steady,
            peak,
            samples: samples.len(),
        }
    }
}

fn merge_engine<'a>(
    readings: impl Iterator<Item = &'a Reading<EngineBytes>>,
) -> Reading<EngineBytes> {
    let mut cpu_bytes = None;
    let mut gpu_bytes = None;
    let mut reasons = Vec::new();
    for reading in readings {
        match reading {
            Reading::Measured(value) => {
                cpu_bytes =
                    Some(cpu_bytes.map_or(value.cpu_bytes, |n: u64| n.max(value.cpu_bytes)));
                gpu_bytes =
                    Some(gpu_bytes.map_or(value.gpu_bytes, |n: u64| n.max(value.gpu_bytes)));
            }
            Reading::Unavailable(reason) => reasons.push(reason.clone()),
        }
    }
    match (cpu_bytes, gpu_bytes) {
        (Some(cpu_bytes), Some(gpu_bytes)) => Reading::Measured(EngineBytes {
            cpu_bytes,
            gpu_bytes,
        }),
        _ => Reading::Unavailable(
            reasons
                .into_iter()
                .next()
                .unwrap_or_else(|| "no measured samples".to_string()),
        ),
    }
}

fn merge_allocator<'a>(
    readings: impl Iterator<Item = &'a Reading<AllocatorBytes>>,
) -> Reading<AllocatorBytes> {
    let mut max = None::<AllocatorBytes>;
    let mut reason = None;
    for reading in readings {
        match reading {
            Reading::Measured(value) => {
                let old = max.get_or_insert_with(|| value.clone());
                old.allocated_bytes = old.allocated_bytes.max(value.allocated_bytes);
                old.reserved_bytes = old.reserved_bytes.max(value.reserved_bytes);
                old.allocations = old.allocations.max(value.allocations);
                old.blocks = old.blocks.max(value.blocks);
            }
            Reading::Unavailable(value) => {
                reason.get_or_insert_with(|| value.clone());
            }
        }
    }
    max.map_or_else(
        || Reading::Unavailable(reason.unwrap_or_else(|| "no measured samples".to_string())),
        Reading::Measured,
    )
}

fn merge_skia<'a>(readings: impl Iterator<Item = &'a Reading<SkiaBudget>>) -> Reading<SkiaBudget> {
    let mut max = None::<SkiaBudget>;
    let mut reason = None;
    for reading in readings {
        match reading {
            Reading::Measured(value) => {
                if max
                    .as_ref()
                    .is_none_or(|current| value.bytes > current.bytes)
                {
                    max = Some(value.clone());
                }
            }
            Reading::Unavailable(value) => {
                reason.get_or_insert_with(|| value.clone());
            }
        }
    }
    max.map_or_else(
        || Reading::Unavailable(reason.unwrap_or_else(|| "no measured samples".to_string())),
        Reading::Measured,
    )
}

fn merge_vk<'a>(readings: impl Iterator<Item = &'a Reading<Vec<VkHeap>>>) -> Reading<Vec<VkHeap>> {
    let mut heaps = Vec::<Option<VkHeap>>::new();
    let mut reason = None;
    for reading in readings {
        match reading {
            Reading::Measured(values) => {
                let len = heaps.len().max(values.len());
                heaps.resize_with(len, || None);
                for (index, value) in values.iter().enumerate() {
                    if heaps[index]
                        .as_ref()
                        .is_none_or(|current| value.usage_bytes > current.usage_bytes)
                    {
                        heaps[index] = Some(value.clone());
                    }
                }
            }
            Reading::Unavailable(value) => {
                reason.get_or_insert_with(|| value.clone());
            }
        }
    }
    if heaps.is_empty() {
        return Reading::Unavailable(reason.unwrap_or_else(|| "no measured samples".to_string()));
    }
    Reading::Measured(heaps.into_iter().flatten().collect())
}

fn merge_process<'a>(
    processes: impl Iterator<Item = &'a ProcessMemory>,
    first: &ProcessMemory,
) -> ProcessMemory {
    let processes: Vec<_> = processes.collect();
    match first {
        ProcessMemory::Linux { .. } => ProcessMemory::Linux {
            rss_bytes: max_u64(processes.iter().filter_map(|p| match p {
                ProcessMemory::Linux { rss_bytes, .. } => Some(rss_bytes),
                _ => None,
            })),
            vk_memory_budget: merge_vk(processes.iter().filter_map(|p| match p {
                ProcessMemory::Linux {
                    vk_memory_budget, ..
                } => Some(vk_memory_budget),
                _ => None,
            })),
        },
        ProcessMemory::Apple { .. } => ProcessMemory::Apple {
            phys_footprint_bytes: max_u64(processes.iter().filter_map(|p| match p {
                ProcessMemory::Apple {
                    phys_footprint_bytes,
                } => Some(phys_footprint_bytes),
                _ => None,
            })),
        },
        ProcessMemory::Android { .. } => ProcessMemory::Android {
            pss_bytes: max_u64(processes.iter().filter_map(|p| match p {
                ProcessMemory::Android { pss_bytes, .. } => Some(pss_bytes),
                _ => None,
            })),
            graphics_bytes: max_u64(processes.iter().filter_map(|p| match p {
                ProcessMemory::Android { graphics_bytes, .. } => Some(graphics_bytes),
                _ => None,
            })),
            gl_mtrack_bytes: max_u64(processes.iter().filter_map(|p| match p {
                ProcessMemory::Android {
                    gl_mtrack_bytes, ..
                } => Some(gl_mtrack_bytes),
                _ => None,
            })),
        },
        ProcessMemory::Other { os } => ProcessMemory::Other { os: os.clone() },
    }
}

fn max_u64<'a>(readings: impl Iterator<Item = &'a Reading<u64>>) -> Reading<u64> {
    let mut max = None;
    let mut reason = None;
    for reading in readings {
        match reading {
            Reading::Measured(value) => {
                max = Some(max.map_or(*value, |current: u64| current.max(*value)));
            }
            Reading::Unavailable(value) => {
                reason.get_or_insert_with(|| value.clone());
            }
        }
    }
    max.map_or_else(
        || Reading::Unavailable(reason.unwrap_or_else(|| "no measured samples".to_string())),
        Reading::Measured,
    )
}

#[cfg(any(
    feature = "cherenkov",
    feature = "cherenkov-vello",
    feature = "vello-classic",
    feature = "vello-hybrid"
))]
pub(crate) fn wgpu_allocator(
    device: &wgpu::Device,
    backend: wgpu::Backend,
) -> Reading<AllocatorBytes> {
    let Some(report) = device.generate_allocator_report() else {
        return Reading::unavailable(format!(
            "wgpu backend {backend:?} returns no allocator report"
        ));
    };
    let allocations = match u64::try_from(report.allocations.len()) {
        Ok(count) => count,
        Err(error) => {
            return Reading::unavailable(format!(
                "wgpu allocation count does not fit in u64: {error}"
            ));
        }
    };
    let blocks = match u64::try_from(report.blocks.len()) {
        Ok(count) => count,
        Err(error) => {
            return Reading::unavailable(format!(
                "wgpu allocator block count does not fit in u64: {error}"
            ));
        }
    };
    Reading::Measured(AllocatorBytes {
        allocated_bytes: report.total_allocated_bytes,
        reserved_bytes: report.total_reserved_bytes,
        allocations,
        blocks,
    })
}

pub(crate) fn skia_budget(api: &'static str, bytes: usize) -> Reading<SkiaBudget> {
    match u64::try_from(bytes) {
        Ok(bytes) => Reading::Measured(SkiaBudget { api, bytes }),
        Err(error) => {
            Reading::unavailable(format!("{api} resource bytes do not fit in u64: {error}"))
        }
    }
}

#[cfg(all(
    target_os = "linux",
    any(
        feature = "cherenkov",
        feature = "cherenkov-vello",
        feature = "vello-classic",
        feature = "vello-hybrid"
    )
))]
pub(crate) fn wgpu_vk_memory_budget(
    device: &wgpu::Device,
    backend: wgpu::Backend,
    adapter_name: &str,
) -> Reading<Vec<VkHeap>> {
    // SAFETY: the guard keeps the HAL device alive while its Vulkan handles
    // are queried; no handle is destroyed or used after the guard drops.
    let Some(hal_device) = (unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }) else {
        return Reading::unavailable(format!(
            "wgpu backend {backend:?} for adapter {adapter_name} is not Vulkan"
        ));
    };
    let instance = hal_device.shared_instance().raw_instance();
    vk_memory_budget(instance, hal_device.raw_physical_device(), adapter_name)
}

#[cfg(all(
    target_os = "linux",
    any(
        feature = "cherenkov",
        feature = "cherenkov-vello",
        feature = "vello-classic",
        feature = "vello-hybrid",
        feature = "skia"
    )
))]
pub(crate) fn ash_vk_memory_budget(
    instance: &ash::Instance,
    physical_device: ash::vk::PhysicalDevice,
    adapter_name: &str,
) -> Reading<Vec<VkHeap>> {
    vk_memory_budget(instance, physical_device, adapter_name)
}

#[cfg(all(not(target_os = "linux"), feature = "skia"))]
pub(crate) fn ash_vk_memory_budget(
    _instance: &ash::Instance,
    _physical_device: ash::vk::PhysicalDevice,
    _adapter_name: &str,
) -> Reading<Vec<VkHeap>> {
    Reading::unavailable("VK_EXT_memory_budget is sampled only on Linux")
}

#[cfg(all(
    target_os = "linux",
    any(
        feature = "cherenkov",
        feature = "cherenkov-vello",
        feature = "vello-classic",
        feature = "vello-hybrid",
        feature = "skia"
    )
))]
fn vk_memory_budget(
    instance: &ash::Instance,
    physical_device: ash::vk::PhysicalDevice,
    adapter_name: &str,
) -> Reading<Vec<VkHeap>> {
    use ash::vk;
    let extensions =
        match unsafe { instance.enumerate_device_extension_properties(physical_device) } {
            Ok(extensions) => extensions,
            Err(error) => {
                return Reading::unavailable(format!(
                    "could not enumerate Vulkan extensions for adapter {adapter_name}: {error}"
                ));
            }
        };
    if !extensions.iter().any(|extension| {
        extension
            .extension_name_as_c_str()
            .is_ok_and(|name| name == ash::ext::memory_budget::NAME)
    }) {
        return Reading::unavailable(format!(
            "adapter {adapter_name} does not expose VK_EXT_memory_budget"
        ));
    }
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    unsafe {
        instance.get_physical_device_memory_properties2(physical_device, &mut properties);
    }
    let memory = properties.memory_properties;
    let heaps = (0..memory.memory_heap_count)
        .map(|index| {
            let index = usize::try_from(index).expect("Vulkan heap index fits usize");
            VkHeap {
                heap: u32::try_from(index).expect("Vulkan heap index fits u32"),
                device_local: memory.memory_heaps[index]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
                usage_bytes: budget.heap_usage[index],
                budget_bytes: budget.heap_budget[index],
            }
        })
        .collect();
    Reading::Measured(heaps)
}

#[cfg(any(
    feature = "cherenkov",
    feature = "cherenkov-vello",
    feature = "vello-classic",
    feature = "vello-hybrid"
))]
#[cfg(not(target_os = "linux"))]
pub(crate) fn wgpu_vk_memory_budget(
    _device: &wgpu::Device,
    _backend: wgpu::Backend,
    _adapter_name: &str,
) -> Reading<Vec<VkHeap>> {
    Reading::unavailable("VK_EXT_memory_budget is sampled only on Linux")
}

fn process_memory(vk_memory_budget: &Reading<Vec<VkHeap>>, detail: SampleDetail) -> ProcessMemory {
    #[cfg(target_os = "linux")]
    {
        let _ = detail;
        let rss_bytes = std::fs::read_to_string("/proc/self/status")
            .map_err(|error| format!("could not read /proc/self/status: {error}"))
            .and_then(|status| parse_linux_rss(&status));
        ProcessMemory::Linux {
            rss_bytes: reading_result(rss_bytes),
            vk_memory_budget: vk_memory_budget.clone(),
        }
    }
    #[cfg(target_os = "android")]
    {
        let _ = vk_memory_budget;
        let pss_bytes = std::fs::read_to_string("/proc/self/smaps_rollup")
            .map_err(|error| format!("could not read /proc/self/smaps_rollup: {error}"))
            .and_then(|status| parse_android_pss(&status));
        let (graphics_bytes, gl_mtrack_bytes) = match detail {
            SampleDetail::Full => android_dumpsys_memory(),
            SampleDetail::Frame => {
                let reason = "dumpsys meminfo sampled at idle, prepare and steady only";
                (Err(reason.to_string()), Err(reason.to_string()))
            }
        };
        ProcessMemory::Android {
            pss_bytes: reading_result(pss_bytes),
            graphics_bytes: reading_result(graphics_bytes),
            gl_mtrack_bytes: reading_result(gl_mtrack_bytes),
        }
    }
    #[cfg(target_vendor = "apple")]
    {
        let _ = (vk_memory_budget, detail);
        ProcessMemory::Apple {
            phys_footprint_bytes: apple_phys_footprint(),
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    {
        let _ = (vk_memory_budget, detail);
        ProcessMemory::Other {
            os: std::env::consts::OS.to_string(),
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn reading_result<T>(result: Result<T, String>) -> Reading<T> {
    result.map_or_else(Reading::Unavailable, Reading::Measured)
}

#[cfg(any(test, target_os = "linux"))]
fn parse_linux_rss(status: &str) -> Result<u64, String> {
    parse_kilobytes(status, "VmRSS:")
        .map_err(|error| format!("could not parse VmRSS in /proc/self/status: {error}"))
}

#[cfg(any(test, target_os = "android"))]
fn parse_android_pss(smaps_rollup: &str) -> Result<u64, String> {
    parse_kilobytes(smaps_rollup, "Pss:")
        .map_err(|error| format!("could not parse Pss in smaps_rollup: {error}"))
}

#[cfg(any(test, target_os = "android", target_os = "linux"))]
fn parse_kilobytes(text: &str, label: &str) -> Result<u64, String> {
    let line = text
        .lines()
        .map(str::trim_start)
        .find(|line| line.starts_with(label))
        .ok_or_else(|| format!("missing {label} row"))?;
    let mut values = line[label.len()..].split_whitespace();
    let value = values
        .next()
        .ok_or_else(|| format!("missing {label} value"))?
        .parse::<u64>()
        .map_err(|error| format!("invalid {label} value: {error}"))?;
    let unit = values
        .next()
        .ok_or_else(|| format!("missing {label} unit"))?;
    if unit != "kB" {
        return Err(format!("unexpected {label} unit {unit:?}"));
    }
    value
        .checked_mul(1024)
        .ok_or_else(|| format!("{label} value overflows bytes"))
}

#[cfg(any(test, target_os = "android"))]
fn parse_android_dumpsys(dumpsys: &str) -> Result<(u64, u64), String> {
    let graphics = parse_dumpsys_kilobytes(dumpsys, "Graphics:")
        .map_err(|error| format!("could not parse Graphics in dumpsys meminfo: {error}"))?;
    let gl_mtrack = parse_dumpsys_kilobytes(dumpsys, "GL mtrack:")
        .map_err(|error| format!("could not parse GL mtrack in dumpsys meminfo: {error}"))?;
    Ok((graphics, gl_mtrack))
}

#[cfg(any(test, target_os = "android"))]
fn parse_dumpsys_kilobytes(text: &str, label: &str) -> Result<u64, String> {
    let line = text
        .lines()
        .map(str::trim_start)
        .find(|line| line.starts_with(label))
        .ok_or_else(|| format!("missing {label} row"))?;
    let value = line[label.len()..]
        .split_whitespace()
        .next()
        .ok_or_else(|| format!("missing {label} value"))?
        .parse::<u64>()
        .map_err(|error| format!("invalid {label} value: {error}"))?;
    value
        .checked_mul(1024)
        .ok_or_else(|| format!("{label} value overflows bytes"))
}

#[cfg(target_os = "android")]
fn android_dumpsys_memory() -> (Result<u64, String>, Result<u64, String>) {
    let output = match std::process::Command::new("dumpsys")
        .args(["meminfo", &std::process::id().to_string()])
        .output()
    {
        Ok(output) if output.status.success() => output,
        Ok(output) => {
            let error = String::from_utf8_lossy(&output.stderr);
            let reason = format!(
                "dumpsys meminfo failed ({}): {}",
                output.status,
                error.trim()
            );
            return (Err(reason.clone()), Err(reason));
        }
        Err(error) => {
            let reason = format!("could not run dumpsys meminfo: {error}");
            return (Err(reason.clone()), Err(reason));
        }
    };
    let text = String::from_utf8_lossy(&output.stdout);
    match parse_android_dumpsys(&text) {
        Ok((graphics, gl_mtrack)) => (Ok(graphics), Ok(gl_mtrack)),
        Err(error) => (Err(error.clone()), Err(error)),
    }
}

#[cfg(target_vendor = "apple")]
#[allow(deprecated)]
fn apple_phys_footprint() -> Reading<u64> {
    use std::mem::size_of;

    #[repr(C)]
    struct TaskVmInfo {
        _virtual_size: u64,
        _region_count: i32,
        _page_size: i32,
        _resident_size: u64,
        _resident_size_peak: u64,
        _device: u64,
        _device_peak: u64,
        _internal: u64,
        _internal_peak: u64,
        _external: u64,
        _external_peak: u64,
        _reusable: u64,
        _reusable_peak: u64,
        _purgeable_volatile_pmap: u64,
        _purgeable_volatile_resident: u64,
        _purgeable_volatile_virtual: u64,
        _compressed: u64,
        _compressed_peak: u64,
        _compressed_lifetime: u64,
        phys_footprint: u64,
    }

    const TASK_VM_INFO: libc::task_flavor_t = 22;
    let mut info = std::mem::MaybeUninit::<TaskVmInfo>::uninit();
    let mut count = u32::try_from(size_of::<TaskVmInfo>() / size_of::<libc::natural_t>())
        .expect("task_vm_info word count fits u32");
    // SAFETY: the output buffer has the task_vm_info rev1 size and is
    // writable for the word count passed to task_info.
    let result = unsafe {
        libc::task_info(
            libc::mach_task_self(),
            TASK_VM_INFO,
            info.as_mut_ptr().cast(),
            &raw mut count,
        )
    };
    if result != libc::KERN_SUCCESS {
        return Reading::unavailable(format!("task_info(TASK_VM_INFO) returned {result}"));
    }
    // SAFETY: KERN_SUCCESS means task_info initialized the output structure.
    Reading::Measured(unsafe { info.assume_init() }.phys_footprint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn snapshot(engine: Reading<EngineBytes>, heaps: Reading<Vec<VkHeap>>) -> MemorySnapshot {
        MemorySnapshot {
            engine,
            wgpu_allocator: Reading::Measured(AllocatorBytes {
                allocated_bytes: 2,
                reserved_bytes: 4,
                allocations: 1,
                blocks: 1,
            }),
            skia_budgeted: Reading::unavailable("not Skia"),
            process: ProcessMemory::Linux {
                rss_bytes: Reading::Measured(100),
                vk_memory_budget: heaps,
            },
        }
    }

    #[test]
    fn reading_serde_shape_is_explicit() {
        assert_eq!(
            serde_json::to_value(Reading::Measured(42)).unwrap(),
            json!({"measured": 42})
        );
        assert_eq!(
            serde_json::to_value(Reading::<u64>::Unavailable("not exposed".into())).unwrap(),
            json!({"unavailable": "not exposed"})
        );
    }

    #[test]
    fn peak_merges_measured_values_and_vk_heaps() {
        let first = snapshot(
            Reading::Measured(EngineBytes {
                cpu_bytes: 8,
                gpu_bytes: 11,
            }),
            Reading::Measured(vec![VkHeap {
                heap: 0,
                device_local: true,
                usage_bytes: 10,
                budget_bytes: 100,
            }]),
        );
        let mut second = snapshot(
            Reading::Measured(EngineBytes {
                cpu_bytes: 6,
                gpu_bytes: 15,
            }),
            Reading::Measured(vec![VkHeap {
                heap: 0,
                device_local: true,
                usage_bytes: 20,
                budget_bytes: 200,
            }]),
        );
        second.wgpu_allocator = Reading::Measured(AllocatorBytes {
            allocated_bytes: 12,
            reserved_bytes: 14,
            allocations: 3,
            blocks: 2,
        });
        second.process = ProcessMemory::Linux {
            rss_bytes: Reading::Measured(200),
            vk_memory_budget: match second.process {
                ProcessMemory::Linux {
                    vk_memory_budget, ..
                } => vk_memory_budget,
                _ => unreachable!(),
            },
        };
        let unavailable = snapshot(
            Reading::unavailable("sample not exposed"),
            Reading::unavailable("extension absent"),
        );
        let peak = MemorySnapshot::peak(&[first, second, unavailable]).unwrap();
        assert_eq!(
            peak.engine,
            Reading::Measured(EngineBytes {
                cpu_bytes: 8,
                gpu_bytes: 15,
            })
        );
        assert_eq!(
            peak.wgpu_allocator,
            Reading::Measured(AllocatorBytes {
                allocated_bytes: 12,
                reserved_bytes: 14,
                allocations: 3,
                blocks: 2,
            })
        );
        let ProcessMemory::Linux {
            rss_bytes,
            vk_memory_budget,
        } = peak.process
        else {
            panic!("expected Linux process readings");
        };
        assert_eq!(
            vk_memory_budget,
            Reading::Measured(vec![VkHeap {
                heap: 0,
                device_local: true,
                usage_bytes: 20,
                budget_bytes: 200,
            }])
        );
        assert_eq!(rss_bytes, Reading::Measured(200));
    }

    #[test]
    fn peak_preserves_unavailable_when_no_value_was_measured() {
        let peak = MemorySnapshot::peak(&[
            snapshot(
                Reading::unavailable("source A"),
                Reading::unavailable("source B"),
            ),
            snapshot(
                Reading::unavailable("source C"),
                Reading::unavailable("source D"),
            ),
        ])
        .unwrap();
        assert_eq!(peak.engine, Reading::unavailable("source A"));
        let ProcessMemory::Linux {
            vk_memory_budget, ..
        } = peak.process
        else {
            panic!("expected Linux process readings");
        };
        assert_eq!(vk_memory_budget, Reading::unavailable("source B"));
    }

    #[test]
    fn parses_linux_rss() {
        let status = "Name:\tbench\nVmRSS:\t1234 kB\n";
        assert_eq!(parse_linux_rss(status).unwrap(), 1_234 * 1024);
        assert!(parse_linux_rss("VmSize:\t9 kB\n").is_err());
    }

    #[test]
    fn parses_android_smaps_rollup_pss() {
        let smaps = "Rss: 100 kB\nPss: 75 kB\n";
        assert_eq!(parse_android_pss(smaps).unwrap(), 75 * 1024);
        assert!(parse_android_pss("Rss: 100 kB\n").is_err());
    }

    #[test]
    fn parses_android_dumpsys_graphics_and_gl_mtrack() {
        let dumpsys = "\
            App Summary\n\
                       Pss(KB)\n\
                    Java Heap: 10\n\
                    Graphics: 20\n\
                    GL mtrack: 30 30 0 0\n";
        assert_eq!(
            parse_android_dumpsys(dumpsys).unwrap(),
            (20 * 1024, 30 * 1024)
        );
        assert!(parse_android_dumpsys("Graphics: 1 kB\n").is_err());
    }
}
