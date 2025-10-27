use approx::relative_eq;
use glam::Vec3A;

/// A triangular face belonging to a [`ConvexHull3d`](crate::dim3::ConvexHull3d).
#[derive(Debug, Clone)]
pub struct TriFace {
    /// The indices of the face's points.
    pub(crate) indices: [u32; 3],
    /// Whether the face is valid (not removed).
    pub(crate) valid: bool,
    /// Whether the points of the face are affinely dependent,
    /// meaning they lie on a single line or point.
    pub(crate) affinely_dependent: bool,
    /// The indices of neighboring faces.
    pub(crate) neighbors: [u32; 3],
    /// The indices in the neighboring faces that point back to this face.
    pub(crate) indirect_neighbors: [u32; 3],
    /// The indices of points in front of the face plane, or the points that can "see" the face.
    pub(crate) outside_points: Vec<u32>,
    /// The normal of the face.
    pub(crate) normal: Vec3A,
    /// The index and distance of the furthest outside point, if any.
    pub(crate) furthest_outside_point: Option<(u32, f32)>,
}

impl TriFace {
    /// Creates a [`Face`] using the `points` with the given `indices`.
    #[inline]
    pub fn from_triangle(points: &[Vec3A], indices: [u32; 3]) -> Self {
        const EPSILON_SQ: f32 = (f32::EPSILON * 100.0) * (f32::EPSILON * 100.0);

        let [a, b, c] = indices.map(|i| points[i as usize]);
        let ab = b - a;
        let ac = c - a;
        let scaled_normal = ab.cross(ac);

        // Check for affinely dependent points (zero-area triangle).
        let affinely_dependent =
            relative_eq!(scaled_normal.length_squared(), 0.0, epsilon = EPSILON_SQ);

        Self {
            indices,
            valid: true,
            affinely_dependent,
            neighbors: [0; 3],
            indirect_neighbors: [0; 3],
            outside_points: Vec::new(),
            normal: scaled_normal.normalize_or_zero(),
            furthest_outside_point: None,
        }
    }

    /// Sets the neighboring faces of the face.
    #[inline]
    pub fn set_neighbors(
        &mut self,
        n0: u32,
        n1: u32,
        n2: u32,
        indirect_n0: u32,
        indirect_n1: u32,
        indirect_n2: u32,
    ) {
        self.neighbors = [n0, n1, n2];
        self.indirect_neighbors = [indirect_n0, indirect_n1, indirect_n2];
    }

    /// Returns the first point index of the edge at the given index.
    #[inline]
    pub fn first_point_from_edge(&self, edge_index: u32) -> u32 {
        self.indices[edge_index as usize]
    }

    /// Returns the second point index of the edge at the given index.
    #[inline]
    pub fn second_point_from_edge(&self, edge_index: u32) -> u32 {
        self.indices[(edge_index as usize + 1) % 3]
    }

    /// Determines whether the face can be "seen" by the given point, in an order-independent manner,
    /// meaning that the result does not depend on the order of the face's points.
    #[inline]
    pub fn can_be_seen_by_point_order_independent(&self, point_i: u32, points: &[Vec3A]) -> bool {
        // If the face is degenerate, it can be seen by any point.
        if self.affinely_dependent {
            return true;
        }

        for i in 0..3 {
            let p0 = points[self.indices[i] as usize];
            let point = points[point_i as usize];
            let distance = (point - p0).dot(self.normal);
            if distance >= 0.0 {
                return true;
            }
        }

        false
    }

    /// Returns the distance to the given point.
    #[inline]
    pub fn distance_to_point(&self, point_i: u32, points: &[Vec3A]) -> f32 {
        let p0 = points[self.indices[0] as usize];
        let point = points[point_i as usize];
        (point - p0).dot(self.normal)
    }

    /// Returns the distance to the given point if the face can be "seen" from it.
    #[inline]
    pub fn distance_to_visible_point(&self, point_i: u32, points: &[Vec3A]) -> Option<f32> {
        // If the face is degenerate, it cannot see any points.
        if self.affinely_dependent {
            return None;
        }

        let distance = self.distance_to_point(point_i, points);
        (distance >= f32::EPSILON * 100.0).then_some(distance)
    }

    /// Adds the given point to the list of outside points for the face.
    #[inline]
    pub fn add_outside_point(&mut self, point_i: u32, points: &[Vec3A]) {
        let distance = self.distance_to_point(point_i, points);
        debug_assert!(distance >= f32::EPSILON);

        if self
            .furthest_outside_point
            .is_none_or(|(_, d)| distance > d)
        {
            self.furthest_outside_point = Some((point_i, distance));
        }

        self.outside_points.push(point_i);
    }

    /// Tries to add the given point to the list of outside points for the face.
    #[inline]
    pub fn try_add_outside_point(&mut self, point_i: u32, points: &[Vec3A]) {
        if let Some(distance) = self.distance_to_visible_point(point_i, points) {
            if self
                .furthest_outside_point
                .is_none_or(|(_, d)| distance > d)
            {
                self.furthest_outside_point = Some((point_i, distance));
            }

            self.outside_points.push(point_i);
        }
    }
}
