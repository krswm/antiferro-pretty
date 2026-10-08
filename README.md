# Pretty-Print Your tenferro Tensors

This crate implements a trait to pretty-print tenferro tensors.

[tenferro](https://github.com/tensor4all/tenferro-rs) is a Rust-native tensor library.

This crate supports the following tenferro tensor structs:

- `tenferro_runtime::TypedTensor`
- `tenferro_runtime::Tensor`

This crate supports the following dtypes:

- `F32`
- `F64`
- `I32`
- `I64`

> [!NOTE]
> This crate is **not** a part of the official tenferro project.

> [!WARNING]
> **Currently, everything of this crate (including public API and git URL) is subject to change.**
>
> Use this crate at your own risk.

License: MIT OR Apache-2.0

*antiferro* is a wordplay of *tenferro* and *antiferromagnetism* (I major condensed matter physics.).
I named it so in order to avoid possible crate name collision in the future with the official `tenferro_*` crates.

## Example

Create a `tenferro_runtime::Tensor` and pretty-print it.

`Cargo.toml`:

```toml
# …

antiferro-pretty = { git = "https://github.com/krswm/antiferro-pretty" }

# …
```

`src/main.rs`:

```rust
// …

use tenferro_runtime::Tensor;

use antiferro_pretty::Pretty;

let tensor = Tensor::from_vec_col_major(vec![20, 10, 2], (0..400).map(|x| x as f64).collect())?;
tensor.show()?;

// …
```

You will see the following in your stdout (terminal).

```
Tensor • dtype: F64 • shape: [20, 10, 2]
│
├╴[:, :, 0]
│ ┌─────────────────────────────────────────────────────────────────────────────┐
│ │ +0.000e0    +2.000e1    +4.000e1     ⋯  +1.400e2    +1.600e2    +1.800e2    │
│ │ +1.000e0    +2.100e1    +4.100e1     ⋯  +1.410e2    +1.610e2    +1.810e2    │
│ │ +2.000e0    +2.200e1    +4.200e1     ⋯  +1.420e2    +1.620e2    +1.820e2    │
│ │      ⋮           ⋮           ⋮               ⋮           ⋮           ⋮      │
│ │ +1.700e1    +3.700e1    +5.700e1     ⋯  +1.570e2    +1.770e2    +1.970e2    │
│ │ +1.800e1    +3.800e1    +5.800e1     ⋯  +1.580e2    +1.780e2    +1.980e2    │
│ │ +1.900e1    +3.900e1    +5.900e1     ⋯  +1.590e2    +1.790e2    +1.990e2    │
│ └─────────────────────────────────────────────────────────────────────────────┘
│
├╴[:, :, 1]
│ ┌─────────────────────────────────────────────────────────────────────────────┐
│ │ +2.000e2    +2.200e2    +2.400e2     ⋯  +3.400e2    +3.600e2    +3.800e2    │
│ │ +2.010e2    +2.210e2    +2.410e2     ⋯  +3.410e2    +3.610e2    +3.810e2    │
│ │ +2.020e2    +2.220e2    +2.420e2     ⋯  +3.420e2    +3.620e2    +3.820e2    │
│ │      ⋮           ⋮           ⋮               ⋮           ⋮           ⋮      │
│ │ +2.170e2    +2.370e2    +2.570e2     ⋯  +3.570e2    +3.770e2    +3.970e2    │
│ │ +2.180e2    +2.380e2    +2.580e2     ⋯  +3.580e2    +3.780e2    +3.980e2    │
│ │ +2.190e2    +2.390e2    +2.590e2     ⋯  +3.590e2    +3.790e2    +3.990e2    │
╵ └─────────────────────────────────────────────────────────────────────────────┘
```

## Credits

- [PyTorch](https://github.com/pytorch/pytorch) and [Julia](https://github.com/JuliaLang/julia) for the inspiration of tensor pretty-printing.
- [tenferro](https://github.com/tensor4all/tenferro-rs) for providing me a great tensor library for Rust.

## Developments

- 2026-10-06: I started working on this crate.

This is a hobby project of mine I started from scratch.

I did **not** use generative AI for this project at all.
