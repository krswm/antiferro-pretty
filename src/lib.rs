// tenferro tensor pretty printed

use std::error::Error;

use tenferro_runtime::Tensor;

#[allow(dead_code)]
fn prettify(_tensor: &Tensor) -> Result<String, Box<dyn Error>> {
    unimplemented!();
}

#[cfg(test)]
mod tests {
    use std::fmt::Write;

    use super::*;

    #[test]
    fn test_77() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![7, 7], (0..49).map(|x| x as f64).collect())?;
        let expected = {
            let mut x = String::new();
            writeln!(x, "7 × 7 F64 Tensor")?;
            writeln!(x, "│")?;
            writeln!(x, "├╴[:, :]")?;
            writeln!(x, "│ ┌─────────────────────────────────────────────────────────────────────────────┐")?;
            writeln!(x, "│ │ +0.000e0    +7.000e0    +1.400e1     ⋯  +2.800e1    +3.500e1    +4.200e1    │")?;
            writeln!(x, "│ │ +1.000e0    +8.000e0    +1.500e1     ⋯  +2.900e1    +3.600e1    +4.300e1    │")?;
            writeln!(x, "│ │ +2.000e0    +9.000e0    +1.600e1     ⋯  +3.000e1    +3.700e1    +4.400e1    │")?;
            writeln!(x, "│ │      ⋮           ⋮           ⋮               ⋮           ⋮           ⋮      │")?;
            writeln!(x, "│ │ +4.000e0    +1.100e0    +1.800e1     ⋯  +3.200e1    +3.900e1    +4.600e1    │")?;
            writeln!(x, "│ │ +5.000e0    +1.200e0    +1.900e1     ⋯  +3.300e1    +4.000e1    +4.700e1    │")?;
            writeln!(x, "│ │ +6.000e0    +1.300e0    +2.000e1     ⋯  +3.400e1    +4.100e1    +4.800e1    │")?;
            writeln!(x, "╵ └─────────────────────────────────────────────────────────────────────────────┘")?;
            //               +X.XXXeXXXX +X.XXXeXXXX +X.XXXeXXXX     +X.XXXeXXXX +X.XXXeXXXX +X.XXXeXXXX
            x
        };
        assert_eq!(prettify(&tensor)?, expected);
        Ok(())
    }
}
