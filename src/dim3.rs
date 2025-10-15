use crate::fixed_hasher::FixedHasher;

use glam::{DMat4, DVec3};
use hashbrown::HashSet;

use std::collections::{BTreeMap, BTreeSet};

/// A triangular face belonging to a [`ConvexHull3d`].
#[derive(Debug, Clone)]
pub struct TriFace {
    /// The indices of the face's points.
    indices: [usize; 3],
    /// The indices of points in front of the face plane, or the points that can "see" the face,
    /// and the distance to each of those points along the normal.
    outside_points: Vec<(usize, f64)>,
    /// The indices of neighboring faces.
    neighbor_faces: Vec<usize>,
    /// The normal of the face.
    normal: DVec3,
    /// How far away from the origin this face is along its normal.
    distance_from_origin: f64,
}

impl TriFace {
    /// Creates a [`Face`] using the `points` with the given `indices`.
    fn from_triangle(points: &[DVec3], indices: [usize; 3]) -> Self {
        let points_of_face = indices.map(|i| points[i]);
        let normal = triangle_normal(points_of_face);
        let origin = normal.dot(points_of_face[0]);

        Self {
            indices,
            outside_points: Vec::new(),
            neighbor_faces: Vec::new(),
            normal,
            distance_from_origin: origin,
        }
    }

    /// Returns the indices of the face's points.
    #[inline]
    pub fn indices(&self) -> [usize; 3] {
        self.indices
    }

    /// Returns a reference to the indices of the face's points.
    #[inline]
    pub fn indices_ref(&self) -> &[usize; 3] {
        &self.indices
    }

    /// Returns the normal of the face.
    #[inline]
    pub fn normal(&self) -> DVec3 {
        self.normal
    }

    /// Returns the distance of the face from the origin along its normal.
    #[inline]
    pub fn distance_from_origin(&self) -> f64 {
        self.distance_from_origin
    }
}

/// An error returned during [`ConvexHull3d`] construction.
#[derive(Debug, Clone, PartialEq)]
pub enum ConvexHull3dError {
    /// The given point set is empty, so no convex hull could be computed.
    Empty,
    /// The convex hull algorithm encountered degeneracies.
    Degenerated,
    /// The given point set cannot produce a valid convex hull.
    DegenerateInput(DegenerateInput),
    /// A round-off error.
    RoundOffError(String),
}

/// The type of degeneracy for when attempting to compute a convex hull for a point set.
#[derive(Debug, Clone, PartialEq)]
pub enum DegenerateInput {
    /// The input points are approximately equal.
    Coincident,
    /// The input points are approximately on the same line.
    Collinear,
    /// The input points are approximately on the same plane.
    Coplanar,
}

impl core::fmt::Display for ConvexHull3dError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self {
            ConvexHull3dError::Empty => write!(f, "empty"),
            ConvexHull3dError::Degenerated => write!(f, "degenerated"),
            ConvexHull3dError::DegenerateInput(kind) => write!(f, "degenerate input: {:?}", kind),
            ConvexHull3dError::RoundOffError(msg) => {
                write!(f, "erroneous results by roundoff error: {}", msg)
            }
        }
    }
}

impl core::error::Error for ConvexHull3dError {}

/// A 3D [convex hull] representing the smallest convex set containing
/// all input points in a given point set.
///
/// This can be thought of as a shrink wrapping of a 3D object.
///
/// [convex hull]: https://en.wikipedia.org/wiki/Convex_hull
///
/// # Example
///
/// ```
/// use glam::DVec3;
/// use quickhull::ConvexHull3d;
///
/// let points = vec![
///     DVec3::new(0.0, 0.0, 0.0),
///     DVec3::new(1.0, 0.0, 0.0),
///     DVec3::new(0.0, 1.0, 0.0),
///     DVec3::new(0.0, 0.0, 1.0),
/// ];
///
/// // No limit on the number of iterations.
/// let max_iter = None;
///
/// // Compute the convex hull.
/// let hull = ConvexHull3d::try_from_points(&points, max_iter).unwrap();
///
/// // Get the vertices and indices of the convex hull.
/// let (vertices, indices) = hull.vertices_indices();
/// ```
#[derive(Clone, Debug)]
pub struct ConvexHull3d {
    /// The points of the convex hull.
    points: Vec<DVec3>,
    /// The faces of the convex hull.
    faces: BTreeMap<usize, TriFace>,
}

/// An iterator over the triangles of a [`ConvexHull3d`].
pub struct TriangleIterator<'a> {
    faces: std::collections::btree_map::Values<'a, usize, TriFace>,
}

impl<'a> Iterator for TriangleIterator<'a> {
    type Item = &'a TriFace;

    fn next(&mut self) -> Option<Self::Item> {
        self.faces.next()
    }
}

impl ConvexHull3d {
    /// Attempts to compute a [`ConvexHull3d`] for the given set of points.
    ///
    /// # Errors
    ///
    /// Returns a [`ConvexHullError`] if hull construction fails.
    pub fn try_from_points(
        points: &[DVec3],
        max_iter: Option<usize>,
    ) -> Result<Self, ConvexHull3dError> {
        let num_points = points.len();

        if num_points == 0 {
            return Err(ConvexHull3dError::Empty);
        }

        if num_points <= 3 {
            return Err(ConvexHull3dError::Degenerated);
        }

        // Create the initial simplex, a tetrahedron in 3D.
        let mut c_hull = Self::init_tetrahedron(points)?;

        // Run the main quick hull algorithm.
        c_hull.update(max_iter)?;

        // Shrink the hull, removing unused points.
        c_hull.remove_unused_points();

        if c_hull.points.len() <= 3 {
            return Err(ConvexHull3dError::Degenerated);
        }

        Ok(c_hull)
    }

    /// Returns the points of the convex hull.
    ///
    /// This consumes the convex hull. If you want a reference to the points,
    /// consider using [`ConvexHull3d::points_ref`] instead.
    #[inline]
    pub fn points(self) -> Vec<DVec3> {
        self.points
    }

    /// Returns a reference to the points of the convex hull.
    #[inline]
    pub fn points_ref(&self) -> &[DVec3] {
        &self.points
    }

    /// Returns the vertices and indices of the convex hull.
    ///
    /// This consumes the convex hull.
    #[inline]
    pub fn vertices_indices(self) -> (Vec<DVec3>, Vec<usize>) {
        let indices = self.compute_indices();
        (self.points, indices)
    }

    /// Returns the indices of the triangles of the convex hull.
    ///
    /// For an iterator over the triangles, consider using [`ConvexHull3d::triangles`] instead.
    #[inline]
    pub fn compute_indices(&self) -> Vec<usize> {
        self.triangles().flat_map(|face| face.indices()).collect()
    }

    /// Returns an iterator over the triangles of the convex hull.
    #[inline]
    #[doc(alias = "faces")]
    pub fn triangles(&self) -> TriangleIterator<'_> {
        TriangleIterator {
            faces: self.faces.values(),
        }
    }

    /// Adds the given points, attempting to update the convex hull.
    ///
    /// # Errors
    ///
    /// Returns a [`ConvexHullError`] if hull construction fails.
    #[inline]
    pub fn add_points(&mut self, points: &[DVec3]) -> Result<(), ConvexHull3dError> {
        self.points.extend_from_slice(points);
        self.update(None)?;
        self.remove_unused_points();

        if self.points.len() <= 3 {
            return Err(ConvexHull3dError::Degenerated);
        }

        Ok(())
    }

    /// Computes the minimum and maximum extents for the given point set, along with
    /// the indices of the minimum and maximum vertices along each coordinate axis.
    fn compute_extremes(points: &[DVec3]) -> ([usize; 3], [usize; 3]) {
        let mut min = points[0];
        let mut max = points[0];

        let mut min_vertices = [0; 3];
        let mut max_vertices = [0; 3];

        for (i, vtx) in points.iter().enumerate().skip(1) {
            if vtx.x < min.x {
                min.x = vtx.x;
                min_vertices[0] = i;
            } else if vtx.x > max.x {
                max.x = vtx.x;
                max_vertices[0] = i;
            }

            if vtx.y < min.y {
                min.y = vtx.y;
                min_vertices[1] = i;
            } else if vtx.y > max.y {
                max.y = vtx.y;
                max_vertices[1] = i;
            }

            if vtx.z < min.z {
                min.z = vtx.z;
                min_vertices[2] = i;
            } else if vtx.z > max.z {
                max.z = vtx.z;
                max_vertices[2] = i;
            }
        }

        (min_vertices, max_vertices)
    }

    fn init_tetrahedron(points: &[DVec3]) -> Result<Self, ConvexHull3dError> {
        let (min_indices, max_indices) = Self::compute_extremes(points);

        // Get the indices of the vertices used for the initial tetrahedron.
        let indices_set = Self::init_tetrahedron_indices(points, min_indices, max_indices)?;

        let mut faces = BTreeMap::new();

        #[allow(clippy::explicit_counter_loop)]
        for i_face in 0..4 {
            let mut face_indices = Vec::new();
            // create face
            for (j, index) in indices_set.iter().enumerate() {
                if j != i_face {
                    face_indices.push(*index);
                }
            }
            let mut face = TriFace::from_triangle(points, face_indices.try_into().unwrap());

            // Check the order of the face's vertices.
            let rem_point = indices_set[i_face];
            let pos = position_from_face(points, &face, rem_point);
            if pos > 0.0 {
                face.indices.swap(0, 1);
                face.normal = -face.normal;
                face.distance_from_origin = -face.distance_from_origin;
            }
            if face.indices.len() != 3 {
                return Err(ConvexHull3dError::RoundOffError(
                    "number of face's vertices should be 3".to_string(),
                ));
            }
            faces.insert(i_face, face);
        }

        // Link neighbors.
        let simplex_face_key: Vec<_> = faces.keys().copied().collect();
        for (key, face) in &mut faces.iter_mut() {
            for neighbors_key in simplex_face_key
                .iter()
                .filter(|neighbor_key| *neighbor_key != key)
            {
                face.neighbor_faces.push(*neighbors_key);
            }
        }

        let simplex = Self {
            points: points.to_vec(),
            faces,
        };

        Ok(simplex)
    }

    /// Computes the indices for the initial tetrahdron built from the given
    /// `points` and the indices of the extreme points along each axis.
    fn init_tetrahedron_indices(
        points: &[DVec3],
        min_indices: [usize; 3],
        max_indices: [usize; 3],
    ) -> Result<[usize; 4], ConvexHull3dError> {
        let mut indices = [0; 4];
        debug_assert!(
            points.len() > 3,
            "This should be checked before this function"
        );

        // The maximum one-dimensional extent of the point-cloud, and the index
        // corresponding to that dimension (x = 0, y = 1, z = 2).
        let mut max_extent = 0.0;
        let mut max_dimension_index = 0;

        for i in 0..3 {
            let extent = points[max_indices[i]][i] - points[min_indices[i]][i];
            if extent > max_extent {
                max_extent = extent;
                max_dimension_index = i;
            }
        }

        if max_extent == 0.0 {
            // The point cloud seems to consist of a single point.
            return Err(ConvexHull3dError::DegenerateInput(
                DegenerateInput::Coincident,
            ));
        }

        // The first two vertices are the ones farthest apart in the maximum dimension.
        indices[0] = max_indices[max_dimension_index];
        indices[1] = min_indices[max_dimension_index];

        // The third vertex should be the one farthest from the line segment
        // between the first two vertices.
        let unit_01 = (points[indices[1]] - points[indices[0]]).normalize();
        let mut normal = DVec3::ZERO;

        let mut max_squared_distance = 0.0;

        for i in 0..points.len() {
            let diff = points[i] - points[indices[0]];
            let cross = unit_01.cross(diff);
            let distance_squared = cross.length_squared();

            if distance_squared > max_squared_distance
                && points[i] != points[indices[0]]
                && points[i] != points[indices[1]]
            {
                max_squared_distance = distance_squared;
                indices[2] = i;
                normal = cross;
            }
        }

        if max_squared_distance == 0.0 {
            return Err(ConvexHull3dError::DegenerateInput(
                DegenerateInput::Collinear,
            ));
        }

        normal = normal.normalize();

        // Recompute the normal to make sure it is perpendicular to unit_10.
        normal = (normal - normal.dot(unit_01) * unit_01).normalize();

        // We now have a base triangle. The fourth vertex should be the one farthest
        // from the triangle along the normal.
        let mut max_distance = 0.0;
        let d0 = points[indices[2]].dot(normal);

        for i in 0..points.len() {
            let distance = (points[i].dot(normal) - d0).abs();

            if distance > max_distance
                && points[i] != points[indices[0]]
                && points[i] != points[indices[1]]
                && points[i] != points[indices[2]]
            {
                max_distance = distance;
                indices[3] = i;
            }
        }

        if max_distance.abs() == 0.0 {
            return Err(ConvexHull3dError::DegenerateInput(
                DegenerateInput::Coplanar,
            ));
        }

        Ok(indices)
    }

    fn update(&mut self, max_iter: Option<usize>) -> Result<(), ConvexHull3dError> {
        let mut face_add_count = *self.faces.keys().last().unwrap() + 1;
        let mut num_iter = 0;
        let mut assigned_point_indices = HashSet::with_hasher(FixedHasher);

        // Mark the points of the faces as assigned.
        for face in self.faces.values() {
            for index in &face.indices {
                assigned_point_indices.insert(*index);
            }
        }

        // Initialize the outside points, sometimes called "conflict lists".
        // They are outside the current hull, but can "see" some faces and therefore could be on the final hull.
        for (_key, face) in &mut self.faces.iter_mut() {
            for (i, _point) in self.points.iter().enumerate() {
                if assigned_point_indices.contains(&i) {
                    continue;
                }

                let pos = position_from_face(&self.points, face, i);

                // If the point can "see" the face, add it to the face's list of outside points.
                if pos > 0.0 {
                    face.outside_points.push((i, pos));
                }
            }
        }

        let (max_iter, truncate) = if let Some(iter) = max_iter {
            (iter, true)
        } else {
            (0, false)
        };

        // The main algorithm of quick hull.
        //
        // For each face that has outside points:
        //
        // 1. Find the outside point that is farthest from the face, the "eye point".
        // 2. Find the "horizon", the vertices that form the boundary between the visible
        //    and non-visible parts of the current hull from the viewpoint of the eye point.
        // 3. Create faces connecting the horizon vertices to the eye point.
        // 4. Assign the orphaned vertices to the new faces, and remove the old faces.
        // 5. Repeat.
        while let Some((key, face)) = self
            .faces
            .iter()
            .find(|(_, face)| !face.outside_points.is_empty())
            .map(|(a, b)| (*a, b))
        {
            if truncate && num_iter >= max_iter {
                break;
            }

            num_iter += 1;

            // Select the furthest point.
            let (furthest_point_index, _) = *face.outside_points.last().unwrap();

            // Initialize the visible set.
            let visible_set =
                initialize_visible_set(&self.points, furthest_point_index, &self.faces, key, face);

            // Get the horizon.
            let horizon = compute_horizon(&visible_set, &self.faces)?;

            // Create new faces connecting the horizon vertices to the furthest point.
            let mut new_keys = Vec::new();
            for (ridge, unvisible) in horizon {
                let mut new_face = vec![furthest_point_index];

                assigned_point_indices.insert(furthest_point_index);

                for point in ridge {
                    new_face.push(point);
                    assigned_point_indices.insert(point);
                }

                if new_face.len() != 3 {
                    return Err(ConvexHull3dError::RoundOffError(
                        "number of new face's vertices should be 3".to_string(),
                    ));
                }

                let mut new_face =
                    TriFace::from_triangle(&self.points, new_face.try_into().unwrap());
                new_face.neighbor_faces.push(unvisible);

                let new_key = face_add_count;
                face_add_count += 1;

                self.faces.insert(new_key, new_face);
                let unvisible_faset = self.faces.get_mut(&unvisible).unwrap();
                unvisible_faset.neighbor_faces.push(new_key);
                new_keys.push(new_key);
            }

            if new_keys.len() < 3 {
                return Err(ConvexHull3dError::RoundOffError(
                    "number of new faces should be grater than 3".to_string(),
                ));
            }

            // Link the faces to their neighbors.
            for (i, key_a) in new_keys.iter().enumerate() {
                let points_of_new_face_a: [usize; 3] = self.faces.get(key_a).unwrap().indices;

                for key_b in new_keys.iter().skip(i + 1) {
                    let points_of_new_face_b: [usize; 3] = self.faces.get(key_b).unwrap().indices;

                    let num_intersection_points = points_of_new_face_a
                        .iter()
                        .filter(|p| points_of_new_face_b.contains(p))
                        .count();

                    if num_intersection_points == 2 {
                        let face_a = self.faces.get_mut(key_a).unwrap();
                        face_a.neighbor_faces.push(*key_b);

                        let face_b = self.faces.get_mut(key_b).unwrap();
                        face_b.neighbor_faces.push(*key_a);
                    }
                }

                let face_a = self.faces.get(key_a).unwrap();
                if face_a.neighbor_faces.len() != 3 {
                    return Err(ConvexHull3dError::RoundOffError(
                        "number of neighbors should be 3".to_string(),
                    ));
                }
            }

            // Check the order of the new face's vertices.
            for new_key in &new_keys {
                let new_face = self.faces.get(new_key).unwrap();
                let mut degenerate = true;

                for assigned_point_index in assigned_point_indices.iter() {
                    let position =
                        position_from_face(&self.points, new_face, *assigned_point_index);

                    if position == 0.0 {
                        continue;
                    } else if position > 0.0 {
                        let new_face = self.faces.get_mut(new_key).unwrap();
                        new_face.indices.swap(0, 1);
                        new_face.normal = -new_face.normal;
                        new_face.distance_from_origin = -new_face.distance_from_origin;
                        degenerate = false;
                        break;
                    }

                    degenerate = false;
                    break;
                }

                if degenerate {
                    return Err(ConvexHull3dError::Degenerated);
                }
            }

            // Assign the orphaned vertices to the new faces.
            let visible_faces_outside_points: Vec<Vec<(usize, f64)>> = visible_set
                .iter()
                .map(|visible| self.faces.get(visible).unwrap().outside_points.clone())
                .collect();

            for new_key in &new_keys {
                let new_face = self.faces.get_mut(new_key).unwrap();
                let mut checked_point_set = HashSet::with_hasher(FixedHasher);

                for outside_points in &visible_faces_outside_points {
                    for (outside_point_index, _) in outside_points.iter() {
                        if assigned_point_indices.contains(outside_point_index)
                            || checked_point_set.contains(outside_point_index)
                        {
                            continue;
                        }

                        checked_point_set.insert(outside_point_index);

                        let pos = position_from_face(&self.points, new_face, *outside_point_index);
                        if pos > 0.0 {
                            new_face.outside_points.push((*outside_point_index, pos));
                        }
                    }
                }

                new_face
                    .outside_points
                    .sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
            }

            // Delete the old visible faces.
            for visible in visible_set.iter().copied() {
                let visible_face_neighbors =
                    self.faces.get(&visible).unwrap().neighbor_faces.clone();
                for neighbor_key in visible_face_neighbors {
                    let neighbor = self.faces.get_mut(&neighbor_key).unwrap();
                    let index = neighbor
                        .neighbor_faces
                        .iter()
                        .enumerate()
                        .find(|(_, k)| **k == visible)
                        .map(|(i, _)| i)
                        .unwrap();
                    neighbor.neighbor_faces.swap_remove(index);
                }
                self.faces.remove(&visible);
            }
        }

        if !self.is_convex() {
            return Err(ConvexHull3dError::RoundOffError("concave".to_string()));
        }

        Ok(())
    }

    fn remove_unused_points(&mut self) {
        // Get the set of unique indices used by the faces.
        let indices_list = BTreeSet::from_iter(self.faces.values().flat_map(|f| f.indices));

        // Map old indices to new indices.
        let indices_list: BTreeMap<usize, usize> = indices_list
            .into_iter()
            .enumerate()
            .map(|(i, index)| (index, i))
            .collect();

        for face in self.faces.values_mut() {
            for i in &mut face.indices {
                *i = *indices_list.get(i).unwrap();
            }
        }

        // Rebuild the points list.
        let mut new_points = Vec::with_capacity(indices_list.len());
        for i in indices_list.keys() {
            new_points.push(self.points[*i]);
        }
        self.points = new_points;
    }

    /// Computes the volume of the convex hull.
    /// Sums up volumes of tetrahedrons from an arbitrary point to all other points
    ///
    /// Returns non-negative value, for extremely small objects might return 0.0
    pub fn volume(&self) -> f64 {
        let indices = self.compute_indices();
        let reference_point = self.points[indices[0]].extend(1.0);
        let mut volume = 0.0;
        for i in (3..indices.len()).step_by(3) {
            let mut mat = DMat4::ZERO;
            for j in 0..3 {
                let row = self.points[indices[i + j]].extend(1.0);
                *mat.col_mut(j) = row;
            }
            *mat.col_mut(3) = reference_point;
            volume += mat.determinant().max(0.0);
        }
        volume / 6.0
    }

    /// Checks if the convex hull is convex with the given tolerance.
    fn is_convex(&self) -> bool {
        for face in self.faces.values() {
            if position_from_face(&self.points, face, 0) > 0.0 {
                return false;
            }
        }
        true
    }

    /// Computes the point on the convex hull that is furthest in the given direction.
    pub fn support_point(&self, direction: DVec3) -> DVec3 {
        let mut max = self.points[0].dot(direction);
        let mut index = 0;

        for (i, point) in self.points.iter().enumerate().skip(1) {
            let dot_product = point.dot(direction);
            if dot_product > max {
                max = dot_product;
                index = i;
            }
        }

        self.points[index]
    }
}

// Computes the indices of the faces that are visible from the point farthest from the given `face`.
fn initialize_visible_set(
    points: &[DVec3],
    furthest_point_index: usize,
    faces: &BTreeMap<usize, TriFace>,
    face_key: usize,
    face: &TriFace,
) -> HashSet<usize, FixedHasher> {
    let mut visible_set = HashSet::with_hasher(FixedHasher);
    visible_set.insert(face_key);
    let mut neighbor_stack: Vec<_> = face.neighbor_faces.to_vec();
    let mut visited_neighbor = HashSet::with_hasher(FixedHasher);
    while let Some(neighbor_key) = neighbor_stack.pop() {
        if visited_neighbor.contains(&neighbor_key) {
            continue;
        }

        visited_neighbor.insert(neighbor_key);

        let neighbor = faces.get(&neighbor_key).unwrap();
        let pos = position_from_face(points, neighbor, furthest_point_index);
        if pos > 0.0 {
            visible_set.insert(neighbor_key);
            neighbor_stack.append(&mut neighbor.neighbor_faces.to_vec());
        }
    }
    visible_set
}

/// Tries to computes the horizon represented as a vector of ridges and the keys of their neighbors.
fn compute_horizon(
    visible_set: &HashSet<usize, FixedHasher>,
    faces: &BTreeMap<usize, TriFace>,
) -> Result<Vec<(Vec<usize>, usize)>, ConvexHull3dError> {
    let mut horizon = Vec::new();
    for visible_key in visible_set.iter() {
        let visible_face = faces.get(visible_key).unwrap();
        let points_of_visible_face: HashSet<_> = visible_face.indices.iter().copied().collect();
        if points_of_visible_face.len() != 3 {
            return Err(ConvexHull3dError::RoundOffError(
                "number of visible face's vertices should be 3".to_string(),
            ));
        }

        for neighbor_key in &visible_face.neighbor_faces {
            // if neighbor is unvisible
            if !visible_set.contains(neighbor_key) {
                let unvisible_neighbor = faces.get(neighbor_key).unwrap();
                let points_of_unvisible_neighbor: HashSet<_> =
                    unvisible_neighbor.indices.iter().copied().collect();
                if points_of_unvisible_neighbor.len() != 3 {
                    return Err(ConvexHull3dError::RoundOffError(
                        "number of unvisible face's vertices should be 3".to_string(),
                    ));
                }

                let horizon_ridge: Vec<_> = points_of_unvisible_neighbor
                    .intersection(&points_of_visible_face)
                    .copied()
                    .collect();
                if horizon_ridge.len() != 2 {
                    return Err(ConvexHull3dError::RoundOffError(
                        "number of ridge's vertices should be 2".to_string(),
                    ));
                }
                horizon.push((horizon_ridge, *neighbor_key));
            }
        }
    }
    if horizon.len() < 3 {
        return Err(ConvexHull3dError::RoundOffError(
            "horizon len < 3".to_string(),
        ));
    }
    Ok(horizon)
}

trait ToRobust {
    fn to_robust(self) -> robust::Coord3D<f64>;
}

impl ToRobust for glam::DVec3 {
    fn to_robust(self) -> robust::Coord3D<f64> {
        let DVec3 { x, y, z } = self;
        robust::Coord3D { x, y, z }
    }
}

fn position_from_face(points: &[DVec3], face: &TriFace, point_index: usize) -> f64 {
    let face_points = face
        .indices
        .iter()
        .copied()
        .map(|i| points[i])
        .collect::<Vec<_>>();

    -robust::orient3d(
        face_points[0].to_robust(),
        face_points[1].to_robust(),
        face_points[2].to_robust(),
        points[point_index].to_robust(),
    )
}

/// Computes the normal of a triangle face with a counterclockwise orientation.
fn triangle_normal([a, b, c]: [DVec3; 3]) -> DVec3 {
    let ab = b - a;
    let ac = c - a;
    ab.cross(ac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_points_coincident() {
        let points = (0..4).map(|_| DVec3::splat(1.0)).collect::<Vec<_>>();

        let result = ConvexHull3d::try_from_points(&points, None);
        assert!(
            matches!(
                result,
                Err(ConvexHull3dError::DegenerateInput(
                    DegenerateInput::Coincident
                ))
            ),
            "{result:?} should be 'coincident' error"
        );
    }

    #[test]
    fn four_points_collinear() {
        let mut points = (0..4).map(|_| DVec3::splat(1.0)).collect::<Vec<_>>();
        points[0].x += f64::EPSILON;
        let result = ConvexHull3d::try_from_points(&points, None);
        assert!(
            matches!(
                result,
                Err(ConvexHull3dError::DegenerateInput(
                    DegenerateInput::Collinear
                ))
            ),
            "{result:?} should be 'collinear' error"
        );
    }

    #[test]
    fn four_points_coplanar() {
        let mut points = (0..4).map(|_| DVec3::splat(1.0)).collect::<Vec<_>>();
        points[0].x += f64::EPSILON;
        points[1].y += f64::EPSILON;
        let result = ConvexHull3d::try_from_points(&points, None);
        assert!(
            matches!(
                result,
                Err(ConvexHull3dError::DegenerateInput(
                    DegenerateInput::Coplanar
                ))
            ),
            "{result:?} should be 'coplanar' error"
        );
    }

    #[test]
    fn four_points_min_volume() {
        let mut points = (0..4).map(|_| DVec3::splat(1.0)).collect::<Vec<_>>();
        points[0].x += 3.0 * f64::EPSILON;
        points[1].y += 3.0 * f64::EPSILON;
        points[2].z += 3.0 * f64::EPSILON;
        let result = ConvexHull3d::try_from_points(&points, None);
        assert_eq!(
            4.3790577010150533e-47,
            result.expect("this should compute ok").volume()
        );
    }

    #[test]
    fn volume_should_be_positive() {
        let mut points = (0..4).map(|_| DVec3::splat(1.0)).collect::<Vec<_>>();
        points[0].x += 1.0 * f64::EPSILON;
        points[1].y += 1.0 * f64::EPSILON;
        points[2].z += 2.0 * f64::EPSILON;
        let result = ConvexHull3d::try_from_points(&points, None);
        assert!(result.expect("this should compute ok").volume() > 0.0);
    }

    #[test]
    fn face_normal_test() {
        let p1 = DVec3::new(-1.0, 0.0, 0.0);
        let p2 = DVec3::new(1.0, 0.0, 0.0);
        let p3 = DVec3::new(0.0, 1.0, 0.0);
        let normal_z = triangle_normal([p1, p2, p3]);
        assert_eq!(normal_z, DVec3::new(0.0, 0.0, 2.0));

        let p1 = DVec3::new(0.0, -1.0, 0.0);
        let p2 = DVec3::new(0.0, 1.0, 0.0);
        let p3 = DVec3::new(0.0, 0.0, 1.0);
        let normal_x = triangle_normal([p1, p2, p3]);
        assert_eq!(normal_x, DVec3::new(2.0, 0.0, 0.0));

        let p1 = DVec3::new(0.0, 0.0, -1.0);
        let p2 = DVec3::new(0.0, 0.0, 1.0);
        let p3 = DVec3::new(1.0, 0.0, 0.0);
        let normal_y = triangle_normal([p1, p2, p3]);
        assert_eq!(normal_y, DVec3::new(0.0, 2.0, 0.0));
    }

    #[test]
    fn inner_outer_test() {
        let p1 = DVec3::new(1.0, 0.0, 0.0);
        let p2 = DVec3::new(0.0, 1.0, 0.0);
        let p3 = DVec3::new(0.0, 0.0, 1.0);
        let outer_point = DVec3::new(0.0, 0.0, 10.0);
        let inner_point = DVec3::new(0.0, 0.0, 0.0);
        let whithin_point = DVec3::new(1.0, 0.0, 0.0);
        let points = vec![p1, p2, p3, outer_point, inner_point, whithin_point];
        let face = TriFace::from_triangle(&points, [0, 1, 2]);
        let outer = position_from_face(&points, &face, 3);
        assert!(outer > 0.0);
        let inner = position_from_face(&points, &face, 4);
        assert!(inner < 0.0);
        let within = position_from_face(&points, &face, 5);
        assert!(within == 0.0);
    }

    #[test]
    fn octahedron_test() {
        let p1 = DVec3::new(1.0, 0.0, 0.0);
        let p2 = DVec3::new(0.0, 1.0, 0.0);
        let p3 = DVec3::new(0.0, 0.0, 1.0);
        let p4 = DVec3::new(-1.0, 0.0, 0.0);
        let p5 = DVec3::new(0.0, -1.0, 0.0);
        let p6 = DVec3::new(0.0, 0.0, -1.0);
        let (_v, i) = ConvexHull3d::try_from_points(&[p1, p2, p3, p4, p5, p6], None)
            .unwrap()
            .vertices_indices();
        assert_eq!(i.len(), 8 * 3);
    }

    #[test]
    fn octahedron_translation_test() {
        let p1 = DVec3::new(1.0, 0.0, 0.0);
        let p2 = DVec3::new(0.0, 1.0, 0.0);
        let p3 = DVec3::new(0.0, 0.0, 1.0);
        let p4 = DVec3::new(-1.0, 0.0, 0.0);
        let p5 = DVec3::new(0.0, -1.0, 0.0);
        let p6 = DVec3::new(0.0, 0.0, -1.0);
        let points: Vec<_> = [p1, p2, p3, p4, p5, p6]
            .into_iter()
            .map(|p| p + DVec3::splat(10.0))
            .collect();
        let (_v, i) = ConvexHull3d::try_from_points(&points, None)
            .unwrap()
            .vertices_indices();
        assert_eq!(i.len(), 8 * 3);
    }

    #[test]
    fn cube_test() {
        let p1 = DVec3::new(1.0, 1.0, 1.0);
        let p2 = DVec3::new(1.0, 1.0, -1.0);
        let p3 = DVec3::new(1.0, -1.0, 1.0);
        let p4 = DVec3::new(1.0, -1.0, -1.0);
        let p5 = DVec3::new(-1.0, 1.0, 1.0);
        let p6 = DVec3::new(-1.0, 1.0, -1.0);
        let p7 = DVec3::new(-1.0, -1.0, 1.0);
        let p8 = DVec3::new(-1.0, -1.0, -1.0);
        let (_v, i) = ConvexHull3d::try_from_points(&[p1, p2, p3, p4, p5, p6, p7, p8], None)
            .unwrap()
            .vertices_indices();
        assert_eq!(i.len(), 6 * 2 * 3);
    }

    #[test]
    fn cube_volume_test() {
        let p1 = DVec3::new(2.0, 2.0, 2.0);
        let p2 = DVec3::new(2.0, 2.0, 0.0);
        let p3 = DVec3::new(2.0, 0.0, 2.0);
        let p4 = DVec3::new(2.0, 0.0, 0.0);
        let p5 = DVec3::new(0.0, 2.0, 2.0);
        let p6 = DVec3::new(0.0, 2.0, 0.0);
        let p7 = DVec3::new(0.0, 0.0, 2.0);
        let p8 = DVec3::new(0.0, 0.0, 0.0);
        let cube = ConvexHull3d::try_from_points(&[p1, p2, p3, p4, p5, p6, p7, p8], None).unwrap();
        assert_eq!(cube.volume(), 8.0);
    }

    // Heavy test (~ 0.75s)
    #[test]
    fn sphere_volume_test() {
        let points = sphere_points(50);
        let hull = ConvexHull3d::try_from_points(&points, None).unwrap();
        let volume = hull.volume();
        let expected_volume = 4.0 / 3.0 * std::f64::consts::PI;
        assert!(
            (volume - expected_volume).abs() < 0.1,
            "Expected {expected_volume}, got {volume}"
        );
    }

    #[test]
    fn cube_support_point_test() {
        let p1 = DVec3::new(1.0, 1.0, 1.0);
        let p2 = DVec3::new(1.0, 1.0, 0.0);
        let p3 = DVec3::new(1.0, 0.0, 1.0);
        let p4 = DVec3::new(1.0, 0.0, 0.0);
        let p5 = DVec3::new(0.0, 1.0, 1.0);
        let p6 = DVec3::new(0.0, 1.0, 0.0);
        let p7 = DVec3::new(0.0, 0.0, 1.0);
        let p8 = DVec3::new(0.0, 0.0, 0.0);
        let cube = ConvexHull3d::try_from_points(&[p1, p2, p3, p4, p5, p6, p7, p8], None).unwrap();
        assert_eq!(cube.support_point(DVec3::splat(0.5)), p1);
    }

    #[test]
    fn flat_test() {
        let p1 = DVec3::new(1.0, 1.0, 10.0);
        let p2 = DVec3::new(1.0, 1.0, 10.0);
        let p3 = DVec3::new(1.0, -1.0, 10.0);
        let p4 = DVec3::new(1.0, -1.0, 10.0);
        let p5 = DVec3::new(-1.0, 1.0, 10.0);
        let p6 = DVec3::new(-1.0, 1.0, 10.0);
        let p7 = DVec3::new(-1.0, -1.0, 10.0);
        let p8 = DVec3::new(-1.0, -1.0, 10.0);
        assert!(
            ConvexHull3d::try_from_points(&[p1, p2, p3, p4, p5, p6, p7, p8], None).is_err_and(
                |err| err == ConvexHull3dError::DegenerateInput(DegenerateInput::Coplanar)
            )
        );
    }

    #[test]
    fn line_test() {
        let points = (0..10)
            .map(|i| DVec3::new(i as f64, 1.0, 10.0))
            .collect::<Vec<_>>();
        assert!(ConvexHull3d::try_from_points(&points, None).is_err_and(
            |err| err == ConvexHull3dError::DegenerateInput(DegenerateInput::Collinear)
        ));
    }

    #[test]
    fn simplex_may_degenerate_test() {
        let points = vec![
            DVec3::new(1.0, 0.0, 1.0),
            DVec3::new(1.0, 1.0, 1.0),
            DVec3::new(2.0, 1.0, 0.0),
            DVec3::new(2.0, 1.0, 1.0),
            DVec3::new(2.0, 0.0, 1.0),
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 2.0),
            DVec3::new(0.0, 1.0, 2.0),
            DVec3::new(0.0, 0.0, 2.0),
            DVec3::new(1.0, 0.0, 2.0),
        ];
        let (_v, _i) = ConvexHull3d::try_from_points(&points, None)
            .unwrap()
            .vertices_indices();
    }

    #[test]
    fn simplex_may_degenerate_test_2() {
        let vertices = vec![
            DVec3::new(0., 0., 0.),
            DVec3::new(1., 0., 0.),
            DVec3::new(1., 0., 1.),
            DVec3::new(0., 0., 1.),
            DVec3::new(0., 1., 0.),
            DVec3::new(1., 1., 0.),
            DVec3::new(1., 1., 1.),
            DVec3::new(0., 1., 1.),
            DVec3::new(2., 1., 0.),
            DVec3::new(2., 1., 1.),
            DVec3::new(2., 0., 1.),
            DVec3::new(2., 0., 0.),
            DVec3::new(1., 1., 2.),
            DVec3::new(0., 1., 2.),
            DVec3::new(0., 0., 2.),
            DVec3::new(1., 0., 2.),
        ];
        let indices = [4, 5, 1, 11, 1, 5, 1, 11, 10, 10, 2, 1, 5, 8, 11];
        let points = indices.iter().map(|i| vertices[*i]).collect::<Vec<_>>();
        let (_v, _i) = ConvexHull3d::try_from_points(&points, None)
            .unwrap()
            .vertices_indices();
    }

    #[cfg(test)]
    fn sphere_points(divisions: usize) -> Vec<DVec3> {
        fn rot_z(point: DVec3, angle: f64) -> DVec3 {
            let e1 = angle.cos() * point[0] - angle.sin() * point[1];
            let e2 = angle.sin() * point[0] + angle.cos() * point[1];
            let e3 = point[2];
            DVec3::new(e1, e2, e3)
        }
        fn rot_x(point: DVec3, angle: f64) -> DVec3 {
            let e1 = point[0];
            let e2 = angle.cos() * point[1] - angle.sin() * point[2];
            let e3 = angle.sin() * point[1] + angle.cos() * point[2];
            DVec3::new(e1, e2, e3)
        }
        let mut points = Vec::new();
        let unit_y = DVec3::Y;
        for step_x in 0..divisions {
            let angle_x = 2.0 * std::f64::consts::PI * (step_x as f64 / divisions as f64);
            let p = rot_x(unit_y, angle_x);
            for step_z in 0..divisions {
                let angle_z = 2.0 * std::f64::consts::PI * (step_z as f64 / divisions as f64);
                let p = rot_z(p, angle_z);
                points.push(p);
            }
        }
        points
    }

    #[test]
    fn sphere_test() {
        let points = sphere_points(10);
        let (_v, _i) = ConvexHull3d::try_from_points(&points, None)
            .unwrap()
            .vertices_indices();
    }

    /// Useful for fuzzing and profiling
    /// creates a sea-urchin like point cloud
    /// with points distributed arbitrarily within a sphere
    #[test]
    fn heavy_sea_urchin_test() {
        use rand::prelude::{Distribution, SeedableRng, SliceRandom};

        // increase this to ~1000 to gather more samples for a sampling profiler
        let iterations = 1;

        for s in 0..iterations {
            let mut rng = rand::rngs::StdRng::seed_from_u64(s);
            let dist = rand::distr::StandardUniform;

            fn rot_z(point: DVec3, angle: f64) -> DVec3 {
                let e1 = angle.cos() * point[0] - angle.sin() * point[1];
                let e2 = angle.sin() * point[0] + angle.cos() * point[1];
                let e3 = point[2];
                DVec3::new(e1, e2, e3)
            }
            fn rot_x(point: DVec3, angle: f64) -> DVec3 {
                let e1 = point[0];
                let e2 = angle.cos() * point[1] - angle.sin() * point[2];
                let e3 = angle.sin() * point[1] + angle.cos() * point[2];
                DVec3::new(e1, e2, e3)
            }
            let mut points = Vec::new();
            let dev = 100;
            let unit_y = DVec3::Y;
            for step_x in 0..dev {
                let angle_x = 2.0 * std::f64::consts::PI * (step_x as f64 / dev as f64);
                let p = rot_x(unit_y, angle_x);
                for step_z in 0..dev {
                    let angle_z = 2.0 * std::f64::consts::PI * (step_z as f64 / dev as f64);
                    let p = rot_z(p, angle_z);
                    let rand_offset: f64 = dist.sample(&mut rng);
                    points.push(p * rand_offset);
                }
            }

            points.shuffle(&mut rng);
            let (_v, _i) = ConvexHull3d::try_from_points(&points, None)
                .unwrap()
                .vertices_indices();
        }
    }
}
