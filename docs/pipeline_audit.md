# Pipeline audit notes

The current implementation reference is [data_flow.md](data_flow.md). These are the central consistency checks:

| Contract | Code | Check |
| --- | --- | --- |
| Ordered raw 9D or 23D state | `common/state.py`, `dataset/processing.py` | Prepared state is 9D TCP pose. |
| Measured trajectory labels | `dataset/processing.py:prepare` | Future TCP positions come from recorded poses, never commands. |
| Independent split | `dataset/processing.py:split_groups` | Group IDs are split before windows. |
| Local origin and C1 joins | `common/spline.py`, upstream `QuadraticSpline` | Start is fixed at zero; control matrix enforces joins. |
| DDPM objective | `training/model.py` | Noise MSE only; U-Net padding excluded. |
| Checkpoint choice | `training/train.py` | Best checkpoint minimizes validation noise MSE. |
| Deployment frame | `deployment/inference.py` | Local XYZ uses latest measured TCP; orientation command stays at first observed orientation. |

Four quadratic segments have 12 control-point vectors. Three C0 and three C1 joins plus the fixed origin leave five free XYZ vectors (15 scalars). Prepared NPZ stores six vectors because its first row is the deterministic zero anchor.

Software tests do not establish simulator tracking, contact, coverage, or sensor freshness. Historical descriptions of 23D **prepared** policy input or on-demand HDF5 viewer image reads no longer describe this code.
