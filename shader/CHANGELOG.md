# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1](https://github.com/water-rs/cherenkov/compare/shader-v0.0.0...shader-v0.0.1) - 2026-09-26

### Added

- *(filtrate,shader)* [**breaking**] image size in the spatial ABI, WGSL libraries, size-relative footprints
- *(filtrate)* [**breaking**] function-form stages composed by cherenkov-shader ([#13](https://github.com/water-rs/cherenkov/pull/13))
- *(shader)* add the naga-IR shader composer ([#9](https://github.com/water-rs/cherenkov/pull/9))

### Fixed

- *(shader)* add the metadata a published crate needs
- *(filtrate)* rotate hue with the CSS/SVG hue-rotate matrix

### Other

- *(release)* publish cherenkov-shader with filtrate
