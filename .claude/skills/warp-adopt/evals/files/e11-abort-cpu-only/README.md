<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# zonecount

Aggregates event streams into per-zone counts for the on-prem analytics
appliance.

## Deployment

Ships as a pure-Python wheel to customer-managed appliances: 8-core x86 boxes,
**no discrete GPU in any SKU**, and no plans for one — the hardware refresh is
locked until 2029. Support policy forbids runtime compilation or any dependency
that needs a compiler toolchain on the appliance.

## Performance today

`assign_zones` is the hot path: for every event it walks the zone list and
tests point-in-polygon, so it is quadratic-ish in the worst case and shows up
as ~70% of a batch run. A batch is 2-5 M events against 200-900 zones and takes
40-90 s on the appliance. Customers have started complaining.

We have never profiled below the function level.
