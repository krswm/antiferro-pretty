use std::error::Error;
use std::fmt::Write;

use tenferro_runtime::{DType, Tensor, TypedTensor};

trait Formattable {
    fn type_as_string(&self) -> String;
    fn dtype_as_string(&self) -> String;
    fn shape(&self) -> &[usize];
    fn value_as_string(&self, offset: usize) -> Result<String, Box<dyn Error>>;
}

macro_rules! impl_formattable_for_typed_tensor {
    ($T:ty, $dtype:literal) => {
        impl Formattable for TypedTensor<$T> {
            fn type_as_string(&self) -> String {
                String::from("TypedTensor")
            }

            fn dtype_as_string(&self) -> String {
                String::from($dtype)
            }

            fn shape(&self) -> &[usize] {
                self.shape()
            }

            fn value_as_string(&self, offset: usize) -> Result<String, Box<dyn Error>> {
                Ok(format!("{:<+11.3e}", self.as_slice()?[offset]))
            }
        }
    };
}

impl_formattable_for_typed_tensor!(f32, "F32");
impl_formattable_for_typed_tensor!(f64, "F64");
impl_formattable_for_typed_tensor!(i32, "I32");
impl_formattable_for_typed_tensor!(i64, "I64");

impl Formattable for Tensor {
    fn type_as_string(&self) -> String {
        String::from("Tensor")
    }

    fn dtype_as_string(&self) -> String {
        format!("{:?}", self.dtype())
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
/// - [`tenferro_runtime::TypedTensor`]
/// - [`tenferro_runtime::Tensor`]
pub trait Pretty {
    /// Pretty-print a tenferro tensor to stdout (terminal).
    ///
    /// Internally, this method uses [`Pretty::format`].
    ///
    /// Supported dtypes are `F32`, `F64`, `I32`, and `I64`.
    ///
    /// # Example
    ///
    /// Create a `tenferro_runtime::Tensor` and pretty-print it.
    ///
    /// ```
    /// use tenferro_runtime::Tensor;
    ///
    /// use antiferro_pretty::Pretty;
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![20, 10, 2], (0..400).map(|x| x as f64).collect()).unwrap();
    /// tensor.print().unwrap();
    /// ```
    fn print(&self) -> Result<(), Box<dyn Error>>;

    /// Prettify a tenferro tensor.
    ///
    /// Supported dtypes are `F32`, `F64`, `I32`, and `I64`.
    ///
    /// # Example
    ///
    /// Create a `tenferro_runtime::Tensor` and obtain its prettified representation.
    ///
    /// ```
    /// use tenferro_runtime::Tensor;
    ///
    /// use antiferro_pretty::Pretty;
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![20, 10, 2], (0..400).map(|x| x as f64).collect()).unwrap();
    /// let expected = String::from(
    ///     "\
    ///     Tensor • dtype: F64 • shape: [20, 10, 2]\n\
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
    /// assert_eq!(tensor.format().unwrap(), expected);
    /// ```
    fn format(&self) -> Result<String, Box<dyn Error>>;
}

impl<T: Formattable> Pretty for T {
    fn print(&self) -> Result<(), Box<dyn Error>> {
        print!("{}", self.format()?);
        Ok(())
    }

    fn format(&self) -> Result<String, Box<dyn Error>> {
        let mut string = String::new();
        format(self, &mut string)?;
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

fn format<T: Formattable>(tensor: &T, string: &mut String) -> Result<(), Box<dyn Error>> {
    writeln!(
        string,
        "{} • dtype: {} • shape: {:?}",
        tensor.type_as_string(),
        tensor.dtype_as_string(),
        tensor.shape()
    )?;

    if tensor.shape().iter().product::<usize>() == 0 {
        return Ok(());
    }

    match tensor.shape().len() {
        0 => {
            format_matrix(tensor, 1, 1, 0, true, string)?;
        }
        1 => {
            format_matrix(tensor, tensor.shape()[0], 1, 0, true, string)?;
        }
        2 => {
            format_matrix(
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

                format_matrix(
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

fn format_matrix<T: Formattable>(
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
    use num_complex::Complex;

    use super::*;

    #[test]
    fn test_constant() {
        let tensor = Tensor::from_vec_col_major(vec![], vec![0.0]).unwrap();
        let expected = String::from(
            "\
            Tensor • dtype: F64 • shape: []\n\
            │ ┌─────────────┐\n\
            │ │ +0.000e0    │\n\
            ╵ └─────────────┘\n\
            ",
            //  +X.XXXeXXXX
        );
        assert_eq!(tensor.format().unwrap(), expected);
    }

    #[test]
    fn test_0() {
        let tensor = Tensor::from_vec_col_major(vec![0], Vec::<f64>::new()).unwrap();
        let expected = String::from("Tensor • dtype: F64 • shape: [0]\n");
        assert_eq!(tensor.format().unwrap(), expected);
    }

    #[test]
    fn test_7() {
        let tensor =
            Tensor::from_vec_col_major(vec![7], (0..7).map(|x| x as f64).collect()).unwrap();
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
        assert_eq!(tensor.format().unwrap(), expected);
    }

    #[test]
    fn test_2() {
        let tensor =
            Tensor::from_vec_col_major(vec![2], (0..2).map(|x| x as f64).collect()).unwrap();
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
        assert_eq!(tensor.format().unwrap(), expected);
    }

    #[test]
    fn test_20() {
        let tensor = Tensor::from_vec_col_major(vec![2, 0], Vec::<f64>::new()).unwrap();
        let expected = String::from("Tensor • dtype: F64 • shape: [2, 0]\n");
        assert_eq!(tensor.format().unwrap(), expected);
    }

    #[test]
    fn test_77() {
        let tensor =
            Tensor::from_vec_col_major(vec![7, 7], (0..49).map(|x| x as f64).collect()).unwrap();
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
        assert_eq!(tensor.format().unwrap(), expected);
    }

    #[test]
    fn test_72() {
        let tensor =
            Tensor::from_vec_col_major(vec![7, 2], (0..14).map(|x| x as f64).collect()).unwrap();
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
        assert_eq!(tensor.format().unwrap(), expected);
    }

    #[test]
    fn test_27() {
        let tensor =
            Tensor::from_vec_col_major(vec![2, 7], (0..14).map(|x| x as f64).collect()).unwrap();
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
        assert_eq!(tensor.format().unwrap(), expected);
    }

    #[test]
    fn test_22() {
        let tensor =
            Tensor::from_vec_col_major(vec![2, 2], (0..4).map(|x| x as f64).collect()).unwrap();
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
        assert_eq!(tensor.format().unwrap(), expected);
    }

    #[test]
    fn test_220() {
        let tensor = Tensor::from_vec_col_major(vec![2, 2, 0], Vec::<f64>::new()).unwrap();
        let expected = String::from("Tensor • dtype: F64 • shape: [2, 2, 0]\n");
        assert_eq!(tensor.format().unwrap(), expected);
    }

    #[test]
    fn test_227() {
        let tensor =
            Tensor::from_vec_col_major(vec![2, 2, 7], (0..28).map(|x| x as f64).collect()).unwrap();
        let expected = String::from(
            "\
            Tensor • dtype: F64 • shape: [2, 2, 7]\n\
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
        assert_eq!(tensor.format().unwrap(), expected);
    }

    #[test]
    fn test_222() {
        let tensor =
            Tensor::from_vec_col_major(vec![2, 2, 2], (0..8).map(|x| x as f64).collect()).unwrap();
        let expected = String::from(
            "\
            Tensor • dtype: F64 • shape: [2, 2, 2]\n\
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
        assert_eq!(tensor.format().unwrap(), expected);
    }

    #[test]
    fn test_2220() {
        let tensor = Tensor::from_vec_col_major(vec![2, 2, 2, 0], Vec::<f64>::new()).unwrap();
        let expected = String::from("Tensor • dtype: F64 • shape: [2, 2, 2, 0]\n");
        assert_eq!(tensor.format().unwrap(), expected);
    }

    #[test]
    fn test_2227() {
        let tensor =
            Tensor::from_vec_col_major(vec![2, 2, 2, 7], (0..56).map(|x| x as f64).collect())
                .unwrap();
        let expected = String::from(
            "\
            Tensor • dtype: F64 • shape: [2, 2, 2, 7]\n\
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
        assert_eq!(tensor.format().unwrap(), expected);
    }

    #[test]
    fn test_2222() {
        let tensor =
            Tensor::from_vec_col_major(vec![2, 2, 2, 2], (0..16).map(|x| x as f64).collect())
                .unwrap();
        let expected = String::from(
            "\
            Tensor • dtype: F64 • shape: [2, 2, 2, 2]\n\
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
        assert_eq!(tensor.format().unwrap(), expected);
    }

    #[test]
    fn test_bool() {
        let tensor =
            Tensor::from_vec_col_major(vec![7], vec![false, true, false, true, false, true, false])
                .unwrap();
        let expected = String::from(
            "\
            Tensor • dtype: BOOL • shape: [7]\n\
            │ ┌─────────────────────────────────────────┐\n\
            │ │ false true  false  ⋯  false true  false │\n\
            ╵ └─────────────────────────────────────────┘\n\
            ",
            //  XXXXX XXXXX XXXXX  ⋯  XXXXX XXXXX XXXXX
        );
        assert_eq!(tensor.format().unwrap(), expected);
    }

    #[test]
    fn test_complex() {
        let tensor = Tensor::from_vec_col_major(
            vec![7],
            vec![
                Complex::<f64>::new(0.0, 1.0),
                Complex::<f64>::new(2.0, 3.0),
                Complex::<f64>::new(4.0, 5.0),
                Complex::<f64>::new(6.0, 7.0),
                Complex::<f64>::new(8.0, 9.0),
                Complex::<f64>::new(10.0, 11.0),
                Complex::<f64>::new(12.0, 13.0),
            ],
        )
        .unwrap();
        let expected = String::from(
            "\
            Tensor • dtype: C64 • shape: [7]\n\
            │ ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐\n\
            │ │       +0.000e0+1.000e0i       +2.000e0+3.000e0i       +4.000e0+5.000e0i  ⋯        +8.000e0+9.000e0i       +1.000e1+1.100e1i       +1.200e1+1.300e1i │\n\
            ╵ └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘\n\
            ",
            //  01234567890123456789012 01234567890123456789012 01234567890123456789012  ⋯  01234567890123456789012 01234567890123456789012 01234567890123456789012
        );
        assert_eq!(tensor.format().unwrap(), expected);
    }

    #[test]
    fn test_typed_tensor() {
        let tensor = TypedTensor::<f64>::from_vec_col_major(vec![], vec![0.0]).unwrap();
        let expected = String::from(
            "\
            TypedTensor • dtype: F64 • shape: []\n\
            │ ┌─────────────┐\n\
            │ │ +0.000e0    │\n\
            ╵ └─────────────┘\n\
            ",
            //  +X.XXXeXXXX
        );
        assert_eq!(tensor.format().unwrap(), expected);
    }
}
