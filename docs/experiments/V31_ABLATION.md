# V3.1 grouped-sampling ablation

## Scope

This is a development ablation, not a hardware-readiness result. It compares
the A*-path-guided `hybrid` baseline and four grouped-sampling settings on BARN
worlds 4, 48, and 49 with seeds 0 and 1, robot radius 0.34 m, `K=1024`,
`T=56`, and `dt=0.05`.

```bash
python3 -m experiments.v31_ablation --worlds 4,48,49 --seeds 2 --jobs 3 \
  --out artifacts/results/milestones/v31_ablation.csv
```

Configurations:

- `baseline`: existing single-Gaussian hybrid.
- `path25`: 25% of samples reserved for the global-path proposal.
- `default`: the former 50% path reserve.
- `path70`: 70% path reserve.
- `conservative`: 50% path reserve, 20-cycle dwell, five-cycle confirmation,
  and a 0.6 switching margin.

An angular sign flip counts only when both signs have magnitude at least
0.05 rad/s. ESS is calculated within every mode before mode selection.

## Aggregate results

| configuration | success | collisions | mean time | mean path | worst clearance | mean sign flips | mean switches | fallback selected | mean ESS | mean path error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 6/6 | 0 | 24.5 s | 9.28 m | 0.098 m | 166.7 | 0.0 | 1.00 | n/a | 0.102 m |
| path25 | 6/6 | 0 | 20.3 s | 8.98 m | 0.095 m | 15.8 | 8.3 | 0.68 | 17.3 | 0.064 m |
| 50% reserve | 6/6 | 0 | 20.3 s | 9.00 m | 0.091 m | 15.5 | 6.7 | 0.73 | 19.3 | 0.066 m |
| path70 | 6/6 | 0 | 20.4 s | 8.99 m | 0.091 m | 17.3 | 6.7 | 0.78 | 19.8 | 0.065 m |
| conservative | 6/6 | 0 | 20.4 s | 8.99 m | 0.083 m | 17.5 | 3.2 | 0.53 | 18.7 | 0.072 m |

## Interpretation

`path25` is the current development default. It retains the best worst-case
clearance among grouped variants, while reducing significant angular sign
reversals by about 90%, reducing mean global-path error by about 37%, and
finishing roughly four seconds faster than baseline. More path samples did not
monotonically improve path tracking or safety. Stronger switching commitment
reduced mode switches but produced the worst clearance and path error, showing
that remaining committed to a degrading mode can be unsafe.

This does **not** establish superiority over baseline:

- Six runs per configuration are too few for statistical claims.
- The grouped worst clearance (0.095 m) is still slightly below baseline
  (0.098 m).
- These worlds share the same BARN environment family.
- There are no dynamic-obstacle, stale-plan, or real skid-steer tests here.

## Blocking observations before V4/hardware

1. Minimum per-mode ESS reached approximately 1 in every grouped setting.
   Mixture importance correction can increase variance further, so V4 must
   include log-domain weights, ESS monitoring, and explicit degeneracy tests.
2. Although only 2–12 mode switches occurred per episode, 6–12 distinct branch
   IDs appeared. V1 identity matching is not stable enough for a long-lived
   Nav2 proposal bank.
3. Grouped field construction averaged about 27 ms, branch proposal generation
   about 6.4 ms, and path proposal generation about 4.2 ms on this workstation.
   The Python loop is not representative of C++, but synchronous construction
   is incompatible with the target 15–20 Hz control loop.
4. The 0.34 m circular footprint and ideal diff-drive model are not yet the
   SLIP footprint and skid-steer dynamics.

## Recommended next work

Before treating V4 as a controller improvement:

1. Improve branch identity matching using overlap/topological signatures rather
   than endpoint distance alone.
2. Add an ESS failure policy and report posterior mass per proposal.
3. Test `path25` on more worlds, dynamic obstacles, and wrong/stale paths.
4. Keep the Nav2 global-path proposal and existing Nav2 critics as the primary
   navigation contract.
5. Implement V4 first as an offline/diagnostic weighting comparison, then
   activate it only if ESS and control continuity remain acceptable.
