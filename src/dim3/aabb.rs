use glam::Vec3A;

/// An Axis-Aligned Bounding Box (AABB) in 3D space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Aabb3d {
    /// The minimum corner of the AABB.
    pub min: Vec3A,
    /// The maximum corner of the AABB.
    pub max: Vec3A,
}

impl Aabb3d {
    /// An invalid (empty) AABB with min set to the maximum possible value
    /// and max set to the minimum possible value.
    pub const INVALID: Self = Self {
        min: Vec3A::splat(f32::MAX),
        max: Vec3A::splat(f32::MIN),
    };

    /// An infinite AABB with min set to negative infinity
    /// and max set to positive infinity.
    pub const LARGEST: Self = Self {
        min: Vec3A::splat(-f32::MAX),
        max: Vec3A::splat(f32::MAX),
    };

    /// An infinite AABB with min set to negative infinity
    /// and max set to positive infinity.
    pub const INFINITY: Self = Self {
        min: Vec3A::splat(-f32::INFINITY),
        max: Vec3A::splat(f32::INFINITY),
    };

    /// Creates a new AABB with the given minimum and maximum points.
    #[inline]
    pub fn new(min: Vec3A, max: Vec3A) -> Self {
        Self { min, max }
    }

    /// Creates a new AABB with both min and max set to the given point.
    #[inline]
    pub fn from_point(point: Vec3A) -> Self {
        Self {
            min: point,
            max: point,
        }
    }

    /// Creates an AABB that bounds the given set of points.
    #[inline]
    pub fn from_points(points: &[Vec3A]) -> Self {
        let mut points = points.iter();
        let mut aabb = Aabb3d::from_point(*points.next().unwrap());
        for point in points {
            aabb.extend(*point);
        }
        aabb
    }

    /// Extends the AABB to include the given point.
    #[inline]
    pub fn extend(&mut self, point: Vec3A) -> &mut Self {
        *self = self.union(&Self::from_point(point));
        self
    }

    /// Returns the union of this AABB and another AABB.
    #[inline]
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Aabb3d {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Returns the intersection of this AABB and another AABB.
    ///
    /// The intersection of two AABBs is the overlapping region that is
    /// common to both AABBs. If the AABBs do not overlap, the resulting
    /// AABB will have min and max values that do not form a valid box
    /// (min will not be less than max).
    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        Aabb3d {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        }
    }

    /// Returns the diagonal vector of the AABB.
    #[inline]
    pub fn diagonal(&self) -> Vec3A {
        self.max - self.min
    }

    /// Returns the center point of the AABB.
    #[inline]
    pub fn center(&self) -> Vec3A {
        (self.max + self.min) * 0.5
    }
}
