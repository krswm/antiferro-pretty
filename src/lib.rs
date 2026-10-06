// tenferro tensor pretty printed

use std::error::Error;

use tenferro_runtime::Tensor;

#[allow(dead_code)]
fn prettify(_tensor: &Tensor) -> Result<String, Box<dyn Error>> {
    unimplemented!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![], vec![0.0])?;
        let expected = String::from(
            "\
            Tensor • dtype: F64 • shape: []\n\
            │ ┌─────────────┐\n\
            │ │ +0.000e0    │\n\
            ╵ └─────────────┘\n\
            ",
            //  +X.XXXeXXXX
        );
        assert_eq!(prettify(&tensor)?, expected);
        Ok(())
    }

    #[test]
    fn test_7() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![7], (0..7).map(|x| x as f64).collect())?;
        let expected = String::from(
            "\
            Tensor • dtype: F64 • shape: [7]\n\
            │ ┌─────────────┐\n\
            │ │ +0.000e0    │\n\
            │ │ +1.000e0    │\n\
            │ │ +2.000e0    │\n\
            │ │      ⋮      │\n\
            │ │ +4.000e0    │\n\
            │ │ +5.000e0    │\n\
            │ │ +6.000e0    │\n\
            ╵ └─────────────┘\n\
            ",
            //  +X.XXXeXXXX
        );
        assert_eq!(prettify(&tensor)?, expected);
        Ok(())
    }

    #[test]
    fn test_2() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![2], (0..2).map(|x| x as f64).collect())?;
        let expected = String::from(
            "\
            Tensor • dtype: F64 • shape: [2]\n\
            │ ┌─────────────┐\n\
            │ │ +0.000e0    │\n\
            │ │ +1.000e0    │\n\
            ╵ └─────────────┘\n\
            ",
            //  +X.XXXeXXXX
        );
        assert_eq!(prettify(&tensor)?, expected);
        Ok(())
    }

    #[test]
    fn test_77() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![7, 7], (0..49).map(|x| x as f64).collect())?;
        let expected = String::from(
            "\
            Tensor • dtype: F64 • shape: [7, 7]\n\
            │ ┌─────────────────────────────────────────────────────────────────────────────┐\n\
            │ │ +0.000e0    +7.000e0    +1.400e1     ⋯  +2.800e1    +3.500e1    +4.200e1    │\n\
            │ │ +1.000e0    +8.000e0    +1.500e1     ⋯  +2.900e1    +3.600e1    +4.300e1    │\n\
            │ │ +2.000e0    +9.000e0    +1.600e1     ⋯  +3.000e1    +3.700e1    +4.400e1    │\n\
            │ │      ⋮           ⋮           ⋮               ⋮           ⋮           ⋮      │\n\
            │ │ +4.000e0    +1.100e1    +1.800e1     ⋯  +3.200e1    +3.900e1    +4.600e1    │\n\
            │ │ +5.000e0    +1.200e1    +1.900e1     ⋯  +3.300e1    +4.000e1    +4.700e1    │\n\
            │ │ +6.000e0    +1.300e1    +2.000e1     ⋯  +3.400e1    +4.100e1    +4.800e1    │\n\
            ╵ └─────────────────────────────────────────────────────────────────────────────┘\n\
            ",
            //  +X.XXXeXXXX +X.XXXeXXXX +X.XXXeXXXX  ⋯  +X.XXXeXXXX +X.XXXeXXXX +X.XXXeXXXX
        );
        assert_eq!(prettify(&tensor)?, expected);
        Ok(())
    }

    #[test]
    fn test_72() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![7, 2], (0..14).map(|x| x as f64).collect())?;
        let expected = String::from(
            "\
            Tensor • dtype: F64 • shape: [7, 2]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +0.000e0    +7.000e0    │\n\
            │ │ +1.000e0    +8.000e0    │\n\
            │ │ +2.000e0    +9.000e0    │\n\
            │ │      ⋮           ⋮      │\n\
            │ │ +4.000e0    +1.100e1    │\n\
            │ │ +5.000e0    +1.200e1    │\n\
            │ │ +6.000e0    +1.300e1    │\n\
            ╵ └─────────────────────────┘\n\
            ",
            //  +X.XXXeXXXX +X.XXXeXXXX
        );
        assert_eq!(prettify(&tensor)?, expected);
        Ok(())
    }

    #[test]
    fn test_27() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![2, 7], (0..14).map(|x| x as f64).collect())?;
        let expected = String::from(
            "\
            Tensor • dtype: F64 • shape: [2, 7]\n\
            │ ┌─────────────────────────────────────────────────────────────────────────────┐\n\
            │ │ +0.000e0    +2.000e0    +4.000e0     ⋯  +8.000e0    +1.000e1    +1.200e1    │\n\
            │ │ +1.000e0    +3.000e0    +5.000e0     ⋯  +9.000e0    +1.100e1    +1.300e1    │\n\
            ╵ └─────────────────────────────────────────────────────────────────────────────┘\n\
            ",
            //  +X.XXXeXXXX +X.XXXeXXXX +X.XXXeXXXX  ⋯  +X.XXXeXXXX +X.XXXeXXXX +X.XXXeXXXX
        );
        assert_eq!(prettify(&tensor)?, expected);
        Ok(())
    }

    #[test]
    fn test_22() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![2, 2], (0..4).map(|x| x as f64).collect())?;
        let expected = String::from(
            "\
            Tensor • dtype: F64 • shape: [2, 2]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +0.000e0    +2.000e0    │\n\
            │ │ +1.000e0    +3.000e0    │\n\
            ╵ └─────────────────────────┘\n\
            ",
            //  +X.XXXeXXXX +X.XXXeXXXX
        );
        assert_eq!(prettify(&tensor)?, expected);
        Ok(())
    }
}
