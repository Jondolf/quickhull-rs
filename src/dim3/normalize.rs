use glam::Vec3A;

/// Normalizes a point cloud by centering it at the origin and scaling it to fit within a cube
/// with side length 2.
pub fn normalize_point_cloud(points: &mut [Vec3A]) {
    let (min, max) = find_min_max(points);
    let diagonal = max - min;
    let max_diagonal_element = diagonal.max_element();
    let center = (min + max) * 0.5;
    let scale = if max_diagonal_element == 0.0 {
        0.0
    } else {
        2.0 / max_diagonal_element
    };

    for p in points.iter_mut() {
        *p = (*p - center) * scale;
    }
}

fn find_min_max(points: &[Vec3A]) -> (Vec3A, Vec3A) {
    let mut min = points[0];
    let mut max = points[0];

    for &p in points.iter().skip(1) {
        min = min.min(p);
        max = max.max(p);
    }

    (min, max)
}
