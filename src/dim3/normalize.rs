use glam::Vec3A;

use crate::dim3::aabb::Aabb3d;

/// Normalizes a point cloud by centering it at the origin and scaling it to fit within a cube
/// with side length 2.
///
/// Returns the AABB of the original point cloud before normalization.
///
/// To recover the original scale and position, use the returned AABB with [`recover_point_cloud`].
pub fn normalize_point_cloud(points: &mut [Vec3A]) -> Aabb3d {
    let aabb = Aabb3d::from_points(points);
    let diagonal = aabb.diagonal();
    let center = aabb.center();
    let scale = 2.0 / diagonal.max_element();

    for p in points.iter_mut() {
        *p = (*p - center) * scale;
    }

    aabb
}

/// Recovers the original scale and position of a normalized point cloud using the provided AABB.
///
/// This is the inverse operation of [`normalize_point_cloud`].
pub fn recover_point_cloud(points: &mut [Vec3A], original_aabb: &Aabb3d) {
    let diagonal = original_aabb.diagonal();
    let center = original_aabb.center();
    let scale = 0.5 * diagonal.max_element();

    for p in points.iter_mut() {
        *p = *p * scale + center;
    }
}
