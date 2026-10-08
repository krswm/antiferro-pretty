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
I named it so in order to avoid possible namespace collision with the official `tenferro-*` packages.

## Example

Create a `tenferro_runtime::Tensor` and pretty-print it.

`Cargo.toml`:

```toml
# …

antiferro-pretty = { git = "https://github.com/krswm/antiferro-pretty.git" }

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

Output:

![Demo](https://raw.githubusercontent.com/krswm/asset/main/antiferro-pretty/demo.png)

## Credits

- [PyTorch](https://github.com/pytorch/pytorch) and [Julia](https://github.com/JuliaLang/julia) for the inspiration of tensor pretty-printing.
- [tenferro](https://github.com/tensor4all/tenferro-rs) for providing me a great tensor library for Rust.

## Developments

- 2026-10-06: I started working on this crate.

This is a hobby project of mine I started from scratch.

I did **not** use generative AI for this project at all.
