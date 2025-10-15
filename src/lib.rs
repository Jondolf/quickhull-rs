//! # Quickhull
//!
//! A Rust-implementation of the Quickhull algorithm for computing convex hulls for point sets.
//!
//! This is a simplified and cleaned up version of [chull](https://github.com/u65xhd/chull),
//! focusing on making the algorithm robust and efficient for the 2D and 3D cases.
//!
//! ## References
//!
//! - C. Bradford Barber et al. 1996. [The Quickhull Algorithm for Convex Hulls](https://www.cise.ufl.edu/~ungor/courses/fall06/papers/QuickHull.pdf) (the original paper)
//! - Dirk Gregorius. GDC 2014. [Physics for Game Programmers: Implementing Quickhull](https://archive.org/details/GDC2014Gregorius)

#![warn(missing_docs)]

mod dim2;
mod dim3;
mod fixed_hasher;

pub use dim2::ConvexHull2d;
pub use dim3::{ConvexHull3d, ConvexHull3dError, DegenerateInput};
