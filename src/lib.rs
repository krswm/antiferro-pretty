use std::error::Error;
use std::fmt::{LowerExp, Write};

use tenferro_runtime::{DType, Tensor, TensorScalar, TypedTensor};

trait Prettifiable {
    fn type_as_string(&self) -> String;
    fn shape(&self) -> &[usize];
    fn value_as_string(&self, offset: usize) -> Result<String, Box<dyn Error>>;
}

impl<T: LowerExp + TensorScalar> Prettifiable for TypedTensor<T> {
    fn type_as_string(&self) -> String {
        String::from("TypedTensor")
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn value_as_string(&self, offset: usize) -> Result<String, Box<dyn Error>> {
        Ok(format!("{:<+11.3e}", self.as_slice()?[offset]))
    }
}

impl Prettifiable for Tensor {
    fn type_as_string(&self) -> String {
        format!("{:?} Tensor", self.dtype())
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn value_as_string(&self, offset: usize) -> Result<String, Box<dyn Error>> {
        match self.dtype() {
            DType::F32 => Ok(format!("{:<+11.3e}", self.as_slice::<f32>()?[offset])),
            DType::F64 => Ok(format!("{:<+11.3e}", self.as_slice::<f64>()?[offset])),
            DType::I32 => Ok(format!("{:<+11.3e}", self.as_slice::<i32>()?[offset])),
            DType::I64 => Ok(format!("{:<+11.3e}", self.as_slice::<i64>()?[offset])),
            dtype => Err(format!("dtype unsupported by pretty: {:?}", dtype).into()),
        }
    }
}

/// Pretty-print your tenferro tensors.
///
/// This trait is implemented for the following structs.
/// - `tenferro_runtime::TypedTensor`
/// - `tenferro_runtime::Tensor`
pub trait Pretty {
    fn show(&self) -> Result<(), Box<dyn Error>>;
    fn prettify(&self) -> Result<String, Box<dyn Error>>;
}

impl<T: Prettifiable> Pretty for T {
    /// Pretty-print a tenferro tensor to stdout (terminal).
    ///
    /// Internally, this method uses [prettify].
    ///
    /// # Example
    ///
    /// Create a `tenferro_runtime::Tensor` and pretty-print it.
    ///
    /// ```
    /// use tenferro_runtime::Tensor;
    ///
    /// use tenferro_pretty::Pretty;
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![20, 10, 2], (0..400).map(|x| x as f64).collect())?;
    /// tensor.show()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn show(&self) -> Result<(), Box<dyn Error>> {
        print!("{}", self.prettify()?);
        Ok(())
    }

    /// Prettify a tenferro tensor.
    ///
    /// # Example
    ///
    /// Create a `tenferro_runtime::Tensor` and obtain its prettified representation.
    ///
    /// ```
    /// use tenferro_runtime::Tensor;
    ///
    /// use tenferro_pretty::Pretty;
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![20, 10, 2], (0..400).map(|x| x as f64).collect())?;
    /// let expected = String::from(
    ///     "\
    ///     F64 Tensor • shape: [20, 10, 2]\n\
    ///     │\n\
    ///     ├╴[:, :, 0]\n\
    ///     │ ┌─────────────────────────────────────────────────────────────────────────────┐\n\
    ///     │ │ +0.000e0    +2.000e1    +4.000e1     ⋯  +1.400e2    +1.600e2    +1.800e2    │\n\
    ///     │ │ +1.000e0    +2.100e1    +4.100e1     ⋯  +1.410e2    +1.610e2    +1.810e2    │\n\
    ///     │ │ +2.000e0    +2.200e1    +4.200e1     ⋯  +1.420e2    +1.620e2    +1.820e2    │\n\
    ///     │ │      ⋮           ⋮           ⋮               ⋮           ⋮           ⋮      │\n\
    ///     │ │ +1.700e1    +3.700e1    +5.700e1     ⋯  +1.570e2    +1.770e2    +1.970e2    │\n\
    ///     │ │ +1.800e1    +3.800e1    +5.800e1     ⋯  +1.580e2    +1.780e2    +1.980e2    │\n\
    ///     │ │ +1.900e1    +3.900e1    +5.900e1     ⋯  +1.590e2    +1.790e2    +1.990e2    │\n\
    ///     │ └─────────────────────────────────────────────────────────────────────────────┘\n\
    ///     │\n\
    ///     ├╴[:, :, 1]\n\
    ///     │ ┌─────────────────────────────────────────────────────────────────────────────┐\n\
    ///     │ │ +2.000e2    +2.200e2    +2.400e2     ⋯  +3.400e2    +3.600e2    +3.800e2    │\n\
    ///     │ │ +2.010e2    +2.210e2    +2.410e2     ⋯  +3.410e2    +3.610e2    +3.810e2    │\n\
    ///     │ │ +2.020e2    +2.220e2    +2.420e2     ⋯  +3.420e2    +3.620e2    +3.820e2    │\n\
    ///     │ │      ⋮           ⋮           ⋮               ⋮           ⋮           ⋮      │\n\
    ///     │ │ +2.170e2    +2.370e2    +2.570e2     ⋯  +3.570e2    +3.770e2    +3.970e2    │\n\
    ///     │ │ +2.180e2    +2.380e2    +2.580e2     ⋯  +3.580e2    +3.780e2    +3.980e2    │\n\
    ///     │ │ +2.190e2    +2.390e2    +2.590e2     ⋯  +3.590e2    +3.790e2    +3.990e2    │\n\
    ///     ╵ └─────────────────────────────────────────────────────────────────────────────┘\n\
    ///     ",
    /// );
    /// assert_eq!(tensor.prettify()?, expected);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn prettify(&self) -> Result<String, Box<dyn Error>> {
        let mut string = String::new();
        prettify(self, &mut string)?;
        Ok(string)
    }
}

const OMIT: usize = usize::MAX;

fn select_indices(num: usize) -> Vec<usize> {
    if num >= 7 {
        vec![0, 1, 2, OMIT, num - 3, num - 2, num - 1]
    } else {
        (0..num).collect()
    }
}

fn prettify<T: Prettifiable>(tensor: &T, string: &mut String) -> Result<(), Box<dyn Error>> {
    writeln!(
        string,
        "{} • shape: {:?}",
        tensor.type_as_string(),
        tensor.shape()
    )?;

    if tensor.shape().iter().product::<usize>() == 0 {
        return Ok(());
    }

    match tensor.shape().len() {
        0 => {
            prettify_matrix(tensor, 1, 1, 0, true, string)?;
        }
        1 => {
            prettify_matrix(tensor, tensor.shape()[0], 1, 0, true, string)?;
        }
        2 => {
            prettify_matrix(
                tensor,
                tensor.shape()[0],
                tensor.shape()[1],
                0,
                true,
                string,
            )?;
        }
        rank => {
            let selected = {
                let mut selected = Vec::new();

                for index in select_indices(tensor.shape()[rank - 1]) {
                    if index != OMIT {
                        selected.push(vec![index]);
                    }
                }

                for num in tensor.shape()[2..(rank - 1)].iter().rev() {
                    let mut new_selected: Vec<Vec<usize>> = Vec::new();
                    for indices in &selected {
                        for index in select_indices(*num) {
                            if index != OMIT {
                                new_selected.push([vec![index], indices.clone()].concat());
                            }
                        }
                    }
                    selected = new_selected;
                }

                selected
            };

            for (i, indices) in selected.iter().enumerate() {
                writeln!(string, "│")?;
                write!(string, "├╴[:, :")?;
                for index in indices {
                    write!(string, ", {index}")?;
                }
                writeln!(string, "]")?;

                let offset = {
                    let mut offset = 0;
                    for (index, num) in
                        std::iter::zip(indices.iter().rev(), tensor.shape()[2..].iter().rev())
                    {
                        offset *= num;
                        offset += index;
                    }
                    offset *= tensor.shape()[1];
                    offset *= tensor.shape()[0];
                    offset
                };

                prettify_matrix(
                    tensor,
                    tensor.shape()[0],
                    tensor.shape()[1],
                    offset,
                    i == selected.len() - 1,
                    string,
                )?;
            }
        }
    }

    Ok(())
}

fn prettify_matrix<T: Prettifiable>(
    tensor: &T,
    num_rows: usize,
    num_cols: usize,
    offset: usize,
    is_last_matrix: bool,
    string: &mut String,
) -> Result<(), Box<dyn Error>> {
    write!(string, "│ ┌")?;
    for col in select_indices(num_cols) {
        if col == OMIT {
            write!(string, "────")?;
        } else {
            write!(string, "────────────")?;
        }
    }
    writeln!(string, "─┐")?;

    for row in select_indices(num_rows) {
        write!(string, "│ │")?;
        if row == OMIT {
            for col in select_indices(num_cols) {
                if col == OMIT {
                    write!(string, "    ")?;
                } else {
                    write!(string, "      ⋮     ")?;
                }
            }
        } else {
            for col in select_indices(num_cols) {
                if col == OMIT {
                    write!(string, "  ⋯ ")?;
                } else {
                    let fine_offset = row + num_rows * col;
                    write!(string, " {}", tensor.value_as_string(offset + fine_offset)?)?;
                }
            }
        }
        writeln!(string, " │")?;
    }

    if is_last_matrix {
        write!(string, "╵ └")?;
    } else {
        write!(string, "│ └")?;
    }
    for col in select_indices(num_cols) {
        if col == OMIT {
            write!(string, "────")?;
        } else {
            write!(string, "────────────")?;
        }
    }
    writeln!(string, "─┘")?;

    Ok(())
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
    fn test_0() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![0], Vec::<f64>::new())?;
        let expected = String::from("F64 Tensor • shape: [0]\n");
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
    fn test_20() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![2, 0], Vec::<f64>::new())?;
        let expected = String::from("F64 Tensor • shape: [2, 0]\n");
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
    fn test_220() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![2, 2, 0], Vec::<f64>::new())?;
        let expected = String::from("F64 Tensor • shape: [2, 2, 0]\n");
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
    fn test_2220() -> Result<(), Box<dyn Error>> {
        let tensor = Tensor::from_vec_col_major(vec![2, 2, 2, 0], Vec::<f64>::new())?;
        let expected = String::from("F64 Tensor • shape: [2, 2, 2, 0]\n");
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
