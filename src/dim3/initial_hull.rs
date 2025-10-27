use approx::{relative_eq, relative_ne};
use glam::{Vec2, Vec3, Vec3A};
use glam_matrix_extras::{SymmetricEigen3, SymmetricMat3};

use crate::{
    dim3::{tri_face::TriFace, validation::validate_face_connectivity},
    ConvexHull2d, ConvexHull3dError, DegenerateInput,
};

/// The initial convex hull structure built from the input points.
///
/// If the input points are degenerate, the hull may be a point, line, or triangle.
pub enum InitialConvexHull3d {
    Point(Vec<Vec3A>, Vec<[u32; 3]>),
    Segment(Vec<Vec3A>, Vec<[u32; 3]>),
    Triangle(Vec<Vec3A>, Vec<[u32; 3]>),
    Tetrahedron(Vec<TriFace>),
}

fn cov(points: &[Vec3A]) -> SymmetricMat3 {
    // Compute the centroid.
    let centroid = points.iter().sum::<Vec3A>() / points.len() as f32;

    // Compute the covariance matrix of the points.
    // TODO: Should we multiply by the reciprocal inside the loop for better numerical stability?
    let mut cov = SymmetricMat3::ZERO;
    for point in points {
        cov += SymmetricMat3::from_outer_product(Vec3::from(*point - centroid));
    }
    cov /= points.len() as f32;
    cov
}

fn degenerate_point_hull(point: Vec3A) -> (Vec<Vec3A>, Vec<[u32; 3]>) {
    (vec![point], vec![[0; 3]; 2])
}

fn degenerate_segment_hull(direction: Vec3A, points: &[Vec3A]) -> (Vec<Vec3A>, Vec<[u32; 3]>) {
    // Find the maximum and minimum projections along the direction.
    let mut min_proj = f32::INFINITY;
    let mut max_proj = f32::NEG_INFINITY;
    let mut min_point = Vec3A::ZERO;
    let mut max_point = Vec3A::ZERO;

    for point in points {
        let proj = direction.dot(*point);
        if proj < min_proj {
            min_proj = proj;
            min_point = *point;
        }
        if proj > max_proj {
            max_proj = proj;
            max_point = *point;
        }
    }

    (vec![min_point, max_point], vec![[0, 1, 0], [1, 0, 0]])
}

pub fn init_tetrahedron(
    points: &[Vec3A],
    normalized_points: &[Vec3A],
    undecided_points: &mut Vec<u32>,
) -> Result<InitialConvexHull3d, ConvexHull3dError> {
    // Compute the eigen decomposition to see if the points are on a lower-dimensional subspace.
    let cov = cov(normalized_points);
    let eig = SymmetricEigen3::new(cov);

    // Count the number of non-zero eigenvalues to determine the rank.
    let rank = eig
        .eigenvalues
        .to_array()
        .iter()
        .filter(|v| relative_ne!(**v, 0.0, epsilon = 1e-7))
        .count();

    match rank {
        0 => {
            // The hull is a single point.
            let (vertices, faces) = degenerate_point_hull(points[0]);
            Ok(InitialConvexHull3d::Point(vertices, faces))
        }
        1 => {
            // The hull is a line segment.
            let direction = Vec3A::from(eig.eigenvectors.z_axis.normalize());
            let (vertices, faces) = degenerate_segment_hull(direction, points);
            Ok(InitialConvexHull3d::Segment(vertices, faces))
        }
        2 => {
            // The hull is a triangle.
            // Project the points onto the plane defined by the two largest eigenvectors.
            let u = Vec3A::from(eig.eigenvectors.x_axis.normalize());
            let v = Vec3A::from(eig.eigenvectors.y_axis.normalize());

            let mut subspace_points: Vec<Vec2> = Vec::with_capacity(normalized_points.len());
            for p in normalized_points {
                subspace_points.push(Vec2::new(p.dot(u), p.dot(v)));
            }

            // Compute the 2D convex hull of the projected points.
            let hull_2d_indices = ConvexHull2d::indices_from_points(&subspace_points);

            // Triangulate the 2D hull to form faces.
            let num_points = hull_2d_indices.len();
            let mut faces = Vec::with_capacity(2 * num_points - 4);

            for i in 1..(num_points - 1) {
                faces.push([0, i as u32, (i + 1) as u32]);
            }

            // Note: The bottom face uses a different starting point to avoid bad topology
            //       where an edge is shared by more than two faces.
            let end = num_points - 1;
            for i in 0..(end - 1) {
                faces.push([end as u32, i as u32 + 1, i as u32]);
            }

            Ok(InitialConvexHull3d::Triangle(
                hull_2d_indices.iter().map(|&i| points[i]).collect(),
                faces,
            ))
        }
        3 => {
            // The hull is a tetrahedron.

            // Find the four points that form the initial tetrahedron.
            let mut indices = [u32::MAX; 4];

            let principal_axis = Vec3A::from(eig.eigenvectors.z_axis);

            // The first two vertices should be the ones farthest apart along the principal axis.
            let mut min_proj = f32::INFINITY;
            let mut max_proj = f32::NEG_INFINITY;
            for (i, point) in points.iter().enumerate() {
                let proj = principal_axis.dot(*point);
                if proj < min_proj {
                    min_proj = proj;
                    indices[0] = i as u32;
                }
                if proj > max_proj {
                    max_proj = proj;
                    indices[1] = i as u32;
                }
            }

            if indices[0] == u32::MAX || indices[1] == u32::MAX {
                return Err(ConvexHull3dError::MissingSupportPoint);
            }

            // The third vertex should be the one farthest from the line segment
            // between the first two vertices.
            let unit_01 = (points[indices[1] as usize] - points[indices[0] as usize]).normalize();

            let mut max_squared_distance = 0.0;

            for i in 0..points.len() {
                let diff = points[i] - points[indices[0] as usize];
                let cross = unit_01.cross(diff);
                let distance_squared = cross.length_squared();

                if distance_squared > max_squared_distance
                    && points[i] != points[indices[0] as usize]
                    && points[i] != points[indices[1] as usize]
                {
                    max_squared_distance = distance_squared;
                    indices[2] = i as u32;
                }
            }

            if indices[2] == u32::MAX {
                return Err(ConvexHull3dError::MissingSupportPoint);
            }

            // Create two faces with opposite normals.
            let mut face1 =
                TriFace::from_triangle(normalized_points, [indices[0], indices[1], indices[2]]);
            let mut face2 =
                TriFace::from_triangle(normalized_points, [indices[1], indices[0], indices[2]]);

            // Link the two faces as neighbors.
            face1.set_neighbors(1, 1, 1, 0, 2, 1);
            face2.set_neighbors(0, 0, 0, 0, 2, 1);

            let mut faces = Vec::with_capacity(4);
            faces.push(face1);
            faces.push(face2);

            // Add outside points to the two faces.
            for (i, &point) in normalized_points.iter().enumerate() {
                if point == normalized_points[indices[0] as usize]
                    || point == normalized_points[indices[1] as usize]
                    || point == normalized_points[indices[2] as usize]
                {
                    continue;
                }

                let mut furthest_face_index = u32::MAX;
                let mut max_distance = 0.0;

                for (j, face) in faces.iter().enumerate() {
                    if let Some(distance) =
                        face.distance_to_visible_point(i as u32, normalized_points)
                    {
                        if distance > max_distance {
                            furthest_face_index = j as u32;
                            max_distance = distance;
                        }
                    }
                }

                if furthest_face_index != u32::MAX {
                    faces[furthest_face_index as usize]
                        .try_add_outside_point(i as u32, normalized_points);
                } else {
                    undecided_points.push(i as u32);
                }

                // If none of the faces can be seen from the point, it is implicitly removed.
            }

            // TODO: Make this optional.
            validate_face_connectivity(0, &faces);
            validate_face_connectivity(1, &faces);

            Ok(InitialConvexHull3d::Tetrahedron(faces))
        }
        _ => unreachable!("Rank can only be 0, 1, 2, or 3."),
    }
}
