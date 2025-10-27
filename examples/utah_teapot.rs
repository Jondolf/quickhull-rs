use bevy::{
    asset::RenderAssetUsages,
    camera::Exposure,
    core_pipeline::tonemapping::Tonemapping,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    scene::SceneInstanceReady,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "App".to_string(),
                ..default()
            }),
            ..default()
        }))
        .add_observer(on_scene_ready)
        .add_systems(Startup, setup)
        .add_systems(Update, rotate)
        .run();
}

#[derive(Component)]
struct Rotating;

fn setup(mut commands: Commands, assets: ResMut<AssetServer>) {
    // Spawn the Utah teapot
    commands.spawn((
        SceneRoot(assets.load(GltfAssetLabel::Scene(0).from_asset("utah_teapot.glb"))),
        Transform::from_xyz(-2.5, 0.0, 0.0).with_scale(Vec3::splat(1.0)),
        Rotating,
    ));

    // Light
    commands.spawn((
        PointLight {
            intensity: 50_000_000.0,
            range: 100.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(1.0, 6.0, 7.0),
    ));

    // Camera
    commands.spawn((
        Camera3d::default(),
        AmbientLight {
            brightness: 4000.0,
            color: Color::WHITE,
            ..default()
        },
        Exposure::SUNLIGHT,
        Tonemapping::AcesFitted,
        Transform::from_xyz(0.0, 2.0, 8.0).looking_at(Vec3::new(0.0, 0.75, 0.0), Vec3::Y),
    ));
}

fn rotate(mut query: Query<&mut Transform, With<Rotating>>, time: Res<Time>) {
    let angle = time.elapsed_secs() * core::f32::consts::PI / 4.0;
    for mut transform in &mut query {
        transform.rotation = Quat::from_rotation_y(angle);
    }
}

fn on_scene_ready(
    ready: On<SceneInstanceReady>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    child_query: Query<&Children>,
    mesh_query: Query<&Mesh3d>,
) {
    // Get all vertices from the loaded scene.
    let mut points = Vec::new();
    for mesh in mesh_query.iter_many(child_query.iter_descendants(ready.entity)) {
        let mesh = meshes.get(mesh.id()).unwrap();
        let vertex_positions = mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
        let vertex_positions = vertex_positions.as_float3().unwrap();
        points.extend(
            vertex_positions
                .iter()
                .map(|&[x, y, z]| Vec3A::new(x, y, z)),
        );
    }

    // Test Parry's convex hull implementation (for comparison).
    let points_na = points
        .iter()
        .map(|p| parry3d::math::Point::new(p.x as f32, p.y as f32, p.z as f32))
        .collect::<Vec<_>>();
    let now = std::time::Instant::now();
    let hull = parry3d::transformation::convex_hull(&points_na);
    info!("Parry computed convex hull in {:.4?}", now.elapsed());
    info!("Parry hull has {} indices", hull.1.len());

    // Compute the convex hull.
    let now = std::time::Instant::now();
    let hull = match quickhull::ConvexHull3d::try_from_points(&points, None) {
        Ok(hull) => hull,
        Err(e) => {
            error!("Failed to compute convex hull: {e}");
            return;
        }
    };
    info!("Computed convex hull in {:.4?}", now.elapsed());

    let (vertices, indices) = hull.vertices_indices();
    let mesh_vertices: Vec<[f32; 3]> = vertices.iter().map(|v| [v.x, v.y, v.z]).collect();
    let mesh_indices: Vec<u32> = indices.iter().flatten().cloned().collect();

    // Create a mesh from the hull.
    let hull_mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, mesh_vertices)
    .with_inserted_indices(Indices::U32(mesh_indices))
    .with_duplicated_vertices()
    .with_computed_flat_normals();

    // Spawn the hull mesh.
    commands.spawn((
        Mesh3d(meshes.add(hull_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.9, 0.9),
            ..default()
        })),
        Transform::from_xyz(2.5, 0.0, 0.0),
        Rotating,
    ));
}
