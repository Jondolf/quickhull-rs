# Quickhull

A Rust implementation of the Quickhull algorithm for computing [convex hulls] for 2D and 3D point sets.

The 3D algorithm was adapted from [chull](https://github.com/u65xhd/chull),
focusing on making it more robust, idiomatic, and efficient.

![The Utah teapot and its convex hull](./images/utah_teapot.png)

*The Utah teapot and its convex hull*

[convex hulls]: https://en.wikipedia.org/wiki/Convex_hull

## 3D Example

```rust
use glam::DVec3;
use quickhull::ConvexHull3d;

let points = vec![
    DVec3::new(0.0, 0.0, 0.0),
    DVec3::new(1.0, 0.0, 0.0),
    DVec3::new(0.0, 1.0, 0.0),
    DVec3::new(0.0, 0.0, 1.0),
];

// No limit on the number of iterations.
let max_iter = None;

// Compute the convex hull.
let hull = ConvexHull3d::try_from_points(&points, max_iter).unwrap();

// Get the vertices and indices of the convex hull.
let (vertices, indices) = hull.vertices_indices();
```

## References

- C. Bradford Barber et al. 1996. [The Quickhull Algorithm for Convex Hulls](https://www.cise.ufl.edu/~ungor/courses/fall06/papers/QuickHull.pdf) (the original paper)
- Dirk Gregorius. GDC 2014. [Physics for Game Programmers: Implementing Quickhull](https://archive.org/details/GDC2014Gregorius)

## License

This Quickhull crate is free and open source. All code in this repository is dual-licensed under either:

- MIT License ([LICENSE-MIT](/LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
- Apache License, Version 2.0 ([LICENSE-APACHE](/LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)

at your option.
