//! This example demonstrates the built-in 3d shapes in Bevy.
//! The scene includes a patterned texture and a rotation for visualizing the normals and UVs.

use std::f32::consts::PI;

use bevy::{
    asset::RenderAssetUsages,
    color::palettes::basic::SILVER,
    mesh::{PrimitiveTopology, VertexAttributeValues},
    pbr::wireframe::{Wireframe, WireframePlugin},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use quickhull::ConvexHull3d;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(ImagePlugin::default_nearest()),
            WireframePlugin::default(),
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (rotate, render_convex_hulls))
        .run();
}

/// A marker component for our shapes so we can query them separately from the ground plane
#[derive(Component)]
struct Shape;

const SHAPES_X_EXTENT: f32 = 14.0;
const EXTRUSION_X_EXTENT: f32 = 16.0;
const Z_EXTENT: f32 = 5.0;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let debug_material = materials.add(StandardMaterial {
        base_color_texture: Some(images.add(uv_debug_texture())),
        ..default()
    });
    let hull_material = materials.add(Color::Srgba(Srgba::new(0.7, 0.7, 0.7, 0.4)));

    let shapes = [
        meshes.add(Cuboid::default()),
        meshes.add(Tetrahedron::default()),
        meshes.add(Capsule3d::default()),
        meshes.add(Torus::default()),
        meshes.add(Cylinder::default()),
        meshes.add(Cone::default()),
        meshes.add(ConicalFrustum::default()),
        meshes.add(Sphere::default().mesh().ico(5).unwrap()),
        meshes.add(Sphere::default().mesh().uv(32, 18)),
    ];

    let extrusions = [
        meshes.add(Extrusion::new(Rectangle::default(), 1.)),
        meshes.add(Extrusion::new(Capsule2d::default(), 1.)),
        meshes.add(Extrusion::new(Annulus::default(), 1.)),
        meshes.add(Extrusion::new(Circle::default(), 1.)),
        meshes.add(Extrusion::new(Ellipse::default(), 1.)),
        meshes.add(Extrusion::new(RegularPolygon::default(), 1.)),
        meshes.add(Extrusion::new(Triangle2d::default(), 1.)),
    ];

    let num_shapes = shapes.len();

    for (i, shape) in shapes.into_iter().enumerate() {
        commands.spawn((
            Mesh3d(shape),
            MeshMaterial3d(debug_material.clone()),
            Transform::from_xyz(
                -SHAPES_X_EXTENT / 2. + i as f32 / (num_shapes - 1) as f32 * SHAPES_X_EXTENT,
                2.0,
                Z_EXTENT / 2.,
            )
            .with_rotation(Quat::from_rotation_x(-PI / 4.)),
            Shape,
            children![(
                Mesh3d(meshes.add(Mesh::new(
                    PrimitiveTopology::default(),
                    RenderAssetUsages::default(),
                ))),
                MeshMaterial3d(hull_material.clone()),
                Transform::from_scale(Vec3::splat(1.1)),
                Wireframe,
            )],
        ));
    }

    let num_extrusions = extrusions.len();

    for (i, shape) in extrusions.into_iter().enumerate() {
        commands.spawn((
            Mesh3d(shape),
            MeshMaterial3d(debug_material.clone()),
            Transform::from_xyz(
                -EXTRUSION_X_EXTENT / 2.
                    + i as f32 / (num_extrusions - 1) as f32 * EXTRUSION_X_EXTENT,
                2.0,
                -Z_EXTENT / 2.,
            )
            .with_rotation(Quat::from_rotation_x(-PI / 4.)),
            Shape,
            children![(
                Mesh3d(meshes.add(Mesh::new(
                    PrimitiveTopology::default(),
                    RenderAssetUsages::default(),
                ))),
                MeshMaterial3d(hull_material.clone()),
                Transform::from_scale(Vec3::splat(1.1)),
                Wireframe,
            )],
        ));
    }

    commands.spawn((
        PointLight {
            shadows_enabled: true,
            intensity: 10_000_000.,
            range: 100.0,
            shadow_depth_bias: 0.2,
            ..default()
        },
        Transform::from_xyz(8.0, 16.0, 8.0),
    ));

    // ground plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(50.0, 50.0).subdivisions(10))),
        MeshMaterial3d(materials.add(Color::from(SILVER))),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 7., 14.0).looking_at(Vec3::new(0., 1., 0.), Vec3::Y),
    ));
}

fn rotate(mut query: Query<&mut Transform, With<Shape>>, time: Res<Time>) {
    let delta_angle = time.delta_secs() * 0.5;
    for mut transform in &mut query {
        transform.rotate_y(delta_angle);
    }
}

/// Creates a colorful test pattern
fn uv_debug_texture() -> Image {
    const TEXTURE_SIZE: usize = 8;

    let mut palette: [u8; 32] = [
        255, 102, 159, 255, 255, 159, 102, 255, 236, 255, 102, 255, 121, 255, 102, 255, 102, 255,
        198, 255, 102, 198, 255, 255, 121, 102, 255, 255, 236, 102, 255, 255,
    ];

    let mut texture_data = [0; TEXTURE_SIZE * TEXTURE_SIZE * 4];
    for y in 0..TEXTURE_SIZE {
        let offset = TEXTURE_SIZE * y * 4;
        texture_data[offset..(offset + TEXTURE_SIZE * 4)].copy_from_slice(&palette);
        palette.rotate_right(4);
    }

    Image::new_fill(
        Extent3d {
            width: TEXTURE_SIZE as u32,
            height: TEXTURE_SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &texture_data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

fn render_convex_hulls(
    query: Query<(&Mesh3d, &Children)>,
    hull_query: Query<&Mesh3d>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    for (handle, children) in &query {
        let Some(mesh) = meshes.get(handle) else {
            continue;
        };

        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            continue;
        };

        let positions = positions
            .iter()
            .map(|v| Vec3A::from(*v))
            .collect::<Vec<_>>();

        let Ok(hull) =
            ConvexHull3d::try_from_points(&positions, None).map_err(|e| warn!("{:?}", e))
        else {
            continue;
        };

        let (vertices, indices) = hull.vertices_indices();

        for handle in hull_query.iter_many(children) {
            let Some(mesh) = meshes.get_mut(handle) else {
                continue;
            };

            mesh.insert_attribute(
                Mesh::ATTRIBUTE_POSITION,
                vertices.iter().map(|v| v.to_array()).collect::<Vec<_>>(),
            );

            mesh.insert_indices(bevy::mesh::Indices::U32(
                indices.iter().flatten().copied().collect::<Vec<_>>(),
            ));

            mesh.duplicate_vertices();

            mesh.compute_flat_normals();
        }
    }
}
