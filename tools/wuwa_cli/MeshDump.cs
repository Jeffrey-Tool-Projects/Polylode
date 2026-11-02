using System.IO.Compression;
using CUE4Parse.UE4.Assets.Exports.Engine;
using CUE4Parse_Conversion.Dto;
using CUE4Parse_Conversion.Options;
using Newtonsoft.Json;
using CUE4Parse.UE4.Assets.Exports.Animation;

internal static class MeshDump
{
    public static void Write(USkinnedAsset mesh, string output)
    {
        mesh.PopulateMorphTargetVerticesData();
        using var dto = new SkeletalMeshDto(mesh, EMeshQuality.Highest);
        using var file = File.Create(Path.Combine(output, mesh.Name + ".source.json.gz"));
        using var gzip = new GZipStream(file, CompressionLevel.Fastest);
        using var text = new StreamWriter(gzip);
        using var writer = new JsonTextWriter(text);
        var document = new
        {
            schema = "wuwa_native_mesh_1121", source = mesh.GetPathName(), unit = "centimeter",
            coordinates = "Unreal left-handed Z-up; no source axis conversion",
            bones = dto.Bones.Select(b => new
            {
                name = b.Name, parent = b.ParentIndex,
                translation = new[] { b.Transform.Translation.X, b.Transform.Translation.Y, b.Transform.Translation.Z },
                rotation_xyzw = new[] { b.Transform.Rotation.X, b.Transform.Rotation.Y, b.Transform.Rotation.Z, b.Transform.Rotation.W },
                scale = new[] { b.Transform.Scale3D.X, b.Transform.Scale3D.Y, b.Transform.Scale3D.Z }
            }),
            materials = dto.Materials,
            morphs = (mesh.MorphTargets ?? []).Select(ptr => ptr.Load<UMorphTarget>())
                .Where(target => target != null).Select(target => new
                {
                    name = target!.Name,
                    lods = target.MorphLODModels.Select(lod => lod.Vertices.Select(v => new
                    {
                        index = v.SourceIdx,
                        delta = new[] { v.PositionDelta.X, v.PositionDelta.Y, v.PositionDelta.Z },
                        normal_delta = new[] { v.TangentZDelta.X, v.TangentZDelta.Y, v.TangentZDelta.Z }
                    }))
                }),
            lods = dto.LODs.Select(l => new
            {
                source_lod = l.SourceLodIndex, indices = l.Indices, sections = l.Sections,
                vertices = l.Vertices.Select(v => new
                {
                    co = new[] { v.Position.X, v.Position.Y, v.Position.Z },
                    no = new[] { v.Normal.X, v.Normal.Y, v.Normal.Z, v.Normal.W },
                    tangent = new[] { v.Tangent.X, v.Tangent.Y, v.Tangent.Z, v.Tangent.W },
                    uv = new[] { v.Uv.U, v.Uv.V }, influences = v.Influences
                }),
                extra_uvs = l.ExtraUvs, colors = l.VertexColors
            })
        };
        JsonSerializer.CreateDefault().Serialize(writer, document);
        Console.WriteLine($"Native mesh {mesh.Name}: {dto.Bones.Length} bones, {dto.LODs.Count} LOD");
    }
}
