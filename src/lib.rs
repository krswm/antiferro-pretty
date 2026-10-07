// tenferro tensor pretty printed

use std::error::Error;
use std::fmt::{LowerExp, Write};

use tenferro_runtime::{DType, Tensor, TensorScalar, TypedTensor};

pub trait Prettifiable {
    fn type_as_string(&self) -> String;
    fn shape(&self) -> &[usize];
    fn get_as_string(&self, indices: &[usize]) -> Result<String, Box<dyn Error>>;
}

impl<T: LowerExp + TensorScalar> Prettifiable for TypedTensor<T> {
    fn type_as_string(&self) -> String {
        String::from("TypedTensor")
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn get_as_string(&self, indices: &[usize]) -> Result<String, Box<dyn Error>> {
        Ok(format!("{:<+11.3e}", *self.get(indices)?))
    }
}

impl Prettifiable for Tensor {
    fn type_as_string(&self) -> String {
        format!("{:?} Tensor", self.dtype())
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn get_as_string(&self, indices: &[usize]) -> Result<String, Box<dyn Error>> {
        match self.dtype() {
            DType::F32 => Ok(format!("{:<+11.3e}", *self.get::<f32>(indices)?)),
            DType::F64 => Ok(format!("{:<+11.3e}", *self.get::<f64>(indices)?)),
            DType::I32 => Ok(format!("{:<+11.3e}", *self.get::<i32>(indices)?)),
            DType::I64 => Ok(format!("{:<+11.3e}", *self.get::<i64>(indices)?)),
            _ => Err(format!("dtype unsupported by prettify: {:?}", self.dtype()).into()),
        }
    }
}

const ELLIPSIS: usize = usize::MAX;

fn select_indices(dim_shape: usize) -> Vec<usize> {
    if dim_shape >= 7 {
        let mut x = Vec::with_capacity(7);
        x.push(0);
        x.push(1);
        x.push(2);
        x.push(ELLIPSIS);
        x.push(dim_shape - 3);
        x.push(dim_shape - 2);
        x.push(dim_shape - 1);
        x
    } else {
        (0..dim_shape).collect()
    }
}

pub trait Prettify {
    fn prettify(&self) -> Result<String, Box<dyn Error>>;
}

impl<T: Prettifiable> Prettify for T {
    fn prettify(&self) -> Result<String, Box<dyn Error>> {
        let mut x = String::new();

        writeln!(
            x,
            "{} • shape: {:?}",
            self.type_as_string(),
            self.shape()
        )?;

        match self.shape().len() {
            0 => {
                writeln!(x, "│ ┌─────────────┐")?;
                writeln!(x, "│ │ {} │", self.get_as_string(&[])?)?;
                writeln!(x, "╵ └─────────────┘")?;
            },
            1 => {
                writeln!(x, "│ ┌─────────────┐")?;
                for i in select_indices(self.shape()[0]) {
                    if i == ELLIPSIS {
                        writeln!(x, "│ │      ⋮      │")?;
                    } else {
                        writeln!(x, "│ │ {} │", self.get_as_string(&[i])?)?;
                    }
                }
                writeln!(x, "╵ └─────────────┘")?;
            },
            2 => {
                write!(x, "│ ┌")?;
                for j in select_indices(self.shape()[1]) {
                    if j == ELLIPSIS {
                        write!(x, "────")?;
                    } else {
                        write!(x, "────────────")?;
                    }
                }
                writeln!(x, "─┐")?;

                for i in select_indices(self.shape()[0]) {
                    write!(x, "│ │")?;
                    if i == ELLIPSIS {
                        for j in select_indices(self.shape()[1]) {
                            if j == ELLIPSIS {
                                write!(x, "    ")?;
                            } else {
                                write!(x, "      ⋮     ")?;
                            }
                        }
                    } else {
                        for j in select_indices(self.shape()[1]) {
                            if j == ELLIPSIS {
                                write!(x, "  ⋯ ")?;
                            } else {
                                write!(x, " {}", self.get_as_string(&[i, j])?)?;
                            }
                        }
                    }
                    writeln!(x, " │")?;
                }

                write!(x, "╵ └")?;
                for j in select_indices(self.shape()[1]) {
                    if j == ELLIPSIS {
                        write!(x, "────")?;
                    } else {
                        write!(x, "────────────")?;
                    }
                }
                writeln!(x, "─┘")?;
                // TDD is powerful
            }
            _ => todo!()
        }

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_tensor() -> Result<(), Box<dyn Error>> {
        let tensor = TypedTensor::<f64>::from_vec_col_major(vec![], vec![0.0])?;
        let expected = String::from(
            "\
            TypedTensor • shape: []\n\
            │ ┌─────────────┐\n\
            │ │ +0.000e0    │\n\
            ╵ └─────────────┘\n\
            ",
            //  +X.XXXeXXXX
        );
        assert_eq!(tensor.prettify()?, expected);
        Ok(())
    }

    #[test]
    fn test_tensor() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![], vec![0.0])?;
        let expected = String::from(
            "\
            F64 Tensor • shape: []\n\
            │ ┌─────────────┐\n\
            │ │ +0.000e0    │\n\
            ╵ └─────────────┘\n\
            ",
            //  +X.XXXeXXXX
        );
        assert_eq!(tensor.prettify()?, expected);
        Ok(())
    }

    #[test]
    fn test_7() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![7], (0..7).map(|x| x as f64).collect())?;
        let expected = String::from(
            "\
            F64 Tensor • shape: [7]\n\
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
        assert_eq!(tensor.prettify()?, expected);
        Ok(())
    }

    #[test]
    fn test_2() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![2], (0..2).map(|x| x as f64).collect())?;
        let expected = String::from(
            "\
            F64 Tensor • shape: [2]\n\
            │ ┌─────────────┐\n\
            │ │ +0.000e0    │\n\
            │ │ +1.000e0    │\n\
            ╵ └─────────────┘\n\
            ",
            //  +X.XXXeXXXX
        );
        assert_eq!(tensor.prettify()?, expected);
        Ok(())
    }

    #[test]
    fn test_77() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![7, 7], (0..49).map(|x| x as f64).collect())?;
        let expected = String::from(
            "\
            F64 Tensor • shape: [7, 7]\n\
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
        assert_eq!(tensor.prettify()?, expected);
        Ok(())
    }

    #[test]
    fn test_72() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![7, 2], (0..14).map(|x| x as f64).collect())?;
        let expected = String::from(
            "\
            F64 Tensor • shape: [7, 2]\n\
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
        assert_eq!(tensor.prettify()?, expected);
        Ok(())
    }

    #[test]
    fn test_27() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![2, 7], (0..14).map(|x| x as f64).collect())?;
        let expected = String::from(
            "\
            F64 Tensor • shape: [2, 7]\n\
            │ ┌─────────────────────────────────────────────────────────────────────────────┐\n\
            │ │ +0.000e0    +2.000e0    +4.000e0     ⋯  +8.000e0    +1.000e1    +1.200e1    │\n\
            │ │ +1.000e0    +3.000e0    +5.000e0     ⋯  +9.000e0    +1.100e1    +1.300e1    │\n\
            ╵ └─────────────────────────────────────────────────────────────────────────────┘\n\
            ",
            //  +X.XXXeXXXX +X.XXXeXXXX +X.XXXeXXXX  ⋯  +X.XXXeXXXX +X.XXXeXXXX +X.XXXeXXXX
        );
        assert_eq!(tensor.prettify()?, expected);
        Ok(())
    }

    #[test]
    fn test_22() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![2, 2], (0..4).map(|x| x as f64).collect())?;
        let expected = String::from(
            "\
            F64 Tensor • shape: [2, 2]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +0.000e0    +2.000e0    │\n\
            │ │ +1.000e0    +3.000e0    │\n\
            ╵ └─────────────────────────┘\n\
            ",
            //  +X.XXXeXXXX +X.XXXeXXXX
        );
        assert_eq!(tensor.prettify()?, expected);
        Ok(())
    }

    #[test]
    fn test_227() -> Result<(), Box<dyn Error>> {
        let tensor =
            Tensor::from_vec_col_major(vec![2, 2, 7], (0..28).map(|x| x as f64).collect())?;
        let expected = String::from(
            "\
            F64 Tensor • shape: [2, 2, 7]\n\
            │\n\
            ├╴[:, :, 0]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +0.000e0    +2.000e0    │\n\
            │ │ +1.000e0    +3.000e0    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 1]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +4.000e0    +6.000e0    │\n\
            │ │ +5.000e0    +7.000e0    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 2]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +8.000e0    +1.000e1    │\n\
            │ │ +9.000e0    +1.100e1    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 4]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +1.600e1    +1.800e1    │\n\
            │ │ +1.700e1    +1.900e1    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 5]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +2.000e1    +2.200e1    │\n\
            │ │ +2.100e1    +2.300e1    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 6]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +2.400e1    +2.600e1    │\n\
            │ │ +2.500e1    +2.700e1    │\n\
            ╵ └─────────────────────────┘\n\
            ",
            //  +X.XXXeXXXX +X.XXXeXXXX
        );
        assert_eq!(tensor.prettify()?, expected);
        Ok(())
    }

    #[test]
    fn test_222() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![2, 2, 2], (0..8).map(|x| x as f64).collect())?;
        let expected = String::from(
            "\
            F64 Tensor • shape: [2, 2, 2]\n\
            │\n\
            ├╴[:, :, 0]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +0.000e0    +2.000e0    │\n\
            │ │ +1.000e0    +3.000e0    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 1]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +4.000e0    +6.000e0    │\n\
            │ │ +5.000e0    +7.000e0    │\n\
            ╵ └─────────────────────────┘\n\
            ",
            //  +X.XXXeXXXX +X.XXXeXXXX
        );
        assert_eq!(tensor.prettify()?, expected);
        Ok(())
    }

    #[test]
    fn test_2227() -> Result<(), Box<dyn Error>> {
        let tensor =
            Tensor::from_vec_col_major(vec![2, 2, 2, 7], (0..56).map(|x| x as f64).collect())?;
        let expected = String::from(
            "\
            F64 Tensor • shape: [2, 2, 2, 7]\n\
            │\n\
            ├╴[:, :, 0, 0]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +0.000e0    +2.000e0    │\n\
            │ │ +1.000e0    +3.000e0    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 1, 0]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +4.000e0    +6.000e0    │\n\
            │ │ +5.000e0    +7.000e0    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 0, 1]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +8.000e0    +1.000e1    │\n\
            │ │ +9.000e0    +1.100e1    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 1, 1]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +1.200e1    +1.400e1    │\n\
            │ │ +1.300e1    +1.500e1    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 0, 2]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +1.600e1    +1.800e1    │\n\
            │ │ +1.700e1    +1.900e1    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 1, 2]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +2.000e1    +2.200e1    │\n\
            │ │ +2.100e1    +2.300e1    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 0, 4]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +3.200e1    +3.400e1    │\n\
            │ │ +3.300e1    +3.500e1    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 1, 4]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +3.600e1    +3.800e1    │\n\
            │ │ +3.700e1    +3.900e1    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 0, 5]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +4.000e1    +4.200e1    │\n\
            │ │ +4.100e1    +4.300e1    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 1, 5]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +4.400e1    +4.600e1    │\n\
            │ │ +4.500e1    +4.700e1    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 0, 6]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +4.800e1    +5.000e1    │\n\
            │ │ +4.900e1    +5.100e1    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 1, 6]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +5.200e1    +5.400e1    │\n\
            │ │ +5.300e1    +5.500e1    │\n\
            ╵ └─────────────────────────┘\n\
            ",
            //  +X.XXXeXXXX +X.XXXeXXXX
        );
        assert_eq!(tensor.prettify()?, expected);
        Ok(())
    }

    #[test]
    fn test_2222() -> Result<(), Box<dyn Error>> {
        let tensor =
            Tensor::from_vec_col_major(vec![2, 2, 2, 2], (0..16).map(|x| x as f64).collect())?;
        let expected = String::from(
            "\
            F64 Tensor • shape: [2, 2, 2, 2]\n\
            │\n\
            ├╴[:, :, 0, 0]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +0.000e0    +2.000e0    │\n\
            │ │ +1.000e0    +3.000e0    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 1, 0]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +4.000e0    +6.000e0    │\n\
            │ │ +5.000e0    +7.000e0    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 0, 1]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +8.000e0    +1.000e1    │\n\
            │ │ +9.000e0    +1.100e1    │\n\
            │ └─────────────────────────┘\n\
            │\n\
            ├╴[:, :, 1, 1]\n\
            │ ┌─────────────────────────┐\n\
            │ │ +1.200e1    +1.400e1    │\n\
            │ │ +1.300e1    +1.500e1    │\n\
            ╵ └─────────────────────────┘\n\
            ",
            //  +X.XXXeXXXX +X.XXXeXXXX
        );
        assert_eq!(tensor.prettify()?, expected);
        Ok(())
    }
}
