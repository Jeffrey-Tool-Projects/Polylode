using CUE4Parse.Compression;
using CUE4Parse.Encryption.Aes;
using CUE4Parse.FileProvider;
using CUE4Parse.UE4.Objects.Core.Misc;
using CUE4Parse.UE4.Versions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using CUE4Parse.UE4.Assets.Exports.Engine;
using CUE4Parse_Conversion;
using CUE4Parse_Conversion.Animations;
using CUE4Parse_Conversion.Options;
using CUE4Parse.UE4.Assets.Exports.Animation;
using CUE4Parse.UE4.Assets.Exports.Material;
using Serilog;
using CUE4Parse.UE4.Assets.Exports.Texture;
using CUE4Parse.UE4.VirtualFileSystem;
using System.Security.Cryptography;

Log.Logger = new LoggerConfiguration().MinimumLevel.Warning().WriteTo.Console().CreateLogger();
CUE4Parse.CUE4ParseLog.UseLogger(Log.Logger);

if (args.Length < 4) throw new ArgumentException("gameRoot keysFile outputDir mode [path]");
var output = Path.GetFullPath(args[2]);
Directory.CreateDirectory(output);
using var provider = new DefaultFileProvider(Path.Combine(args[0], "Client/Content"),
    SearchOption.AllDirectories, new VersionContainer(EGame.GAME_WutheringWaves),
    StringComparer.OrdinalIgnoreCase);
provider.Initialize();
provider.Mount();
var keys = JObject.Parse(File.ReadAllText(args[1]));
var submitted = new Dictionary<FGuid, FAesKey>
{
    [new FGuid()] = new FAesKey(keys.Value<string>("mainKey")!)
};
foreach (var key in keys["dynamicKeys"]!)
    submitted[new FGuid(key.Value<string>("guid")!.Replace("-", ""))] = new FAesKey(key.Value<string>("key")!);
provider.SubmitKeys(submitted);
Console.WriteLine($"Mounted={provider.MountedVfs.Count} Unmounted={provider.UnloadedVfs.Count} Files={provider.Files.Count}");
var report = new
{
    game = "WutheringWaves", parserVersion = typeof(DefaultFileProvider).Assembly.FullName,
    mounted = provider.MountedVfs.Select(v => new { v.Path, count = v.Files.Count }).OrderBy(v => v.Path),
    unmounted = provider.UnloadedVfs.Select(v => new { v.Path, guid = v.EncryptionKeyGuid.ToString() })
};
File.WriteAllText(Path.Combine(output, "mount.json"), JsonConvert.SerializeObject(report, Formatting.Indented));
if (args[3] == "index")
{
    File.WriteAllLines(Path.Combine(output, "files.txt"), provider.Files.Keys.Order());
    return;
}
if (args.Length < 5) throw new ArgumentException("path is required");
OodleHelper.Initialize();
ZlibHelper.Initialize();
var detex = Path.Combine(AppContext.BaseDirectory, "Detex.dll");
CUE4Parse_Conversion.Textures.BC.DetexHelper.LoadDll(detex);
CUE4Parse_Conversion.Textures.BC.DetexHelper.Initialize(detex);
if (args[3] is "raw" or "raw-list")
{
    var paths = args[3] == "raw-list" ? File.ReadAllLines(args[4]).ToHashSet(StringComparer.OrdinalIgnoreCase) : null;
    foreach (var file in provider.Files.Values.Where(f => paths != null ? paths.Contains(f.Path) : f.Path.Contains(args[4], StringComparison.OrdinalIgnoreCase)))
    {
        var destination = Path.GetFullPath(Path.Combine(output, file.Path));
        if (!destination.StartsWith(output + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("Archive path escaped output");
        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        File.WriteAllBytes(destination, file.Read());
        Console.WriteLine(file.Path);
    }
    return;
}
if (args[3] == "json")
{
    var package = provider.LoadPackage(args[4]);
    File.WriteAllText(Path.Combine(output, "package.json"), JsonConvert.SerializeObject(package.GetExports(), Formatting.Indented));
    return;
}
if (args[3] == "blueprints")
{
    var scope = JObject.Parse(File.ReadAllText(args[4]));
    var modelIds = scope["models"]!.SelectMany(model => model["owners"]!)
        .Select(owner => owner.Value<int>("model_id")).Distinct();
    var pending = new Queue<string>();
    foreach (var id in modelIds)
    {
        var preload = scope["preloads"]![id.ToString()]!;
        foreach (var path in preload["actorclass"]!.Values<string>().Append(preload.Value<string>("actorclasspath")))
            if (!string.IsNullOrWhiteSpace(path) && path != "None") pending.Enqueue(path!.Split('.')[0]);
    }
    var visited = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    var records = new List<object>();
    while (pending.TryDequeue(out var path))
    {
        path = provider.FixPath(path).Replace(".uasset", "");
        if (!visited.Add(path)) continue;
        if (!provider.TryGetGameFile(path + ".uasset", out var file)) continue;
        var objects = JArray.Parse(JsonConvert.SerializeObject(provider.LoadPackage(path).GetExports()));
        var name = Path.GetFileName(path) + "_" + Convert.ToHexStringLower(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(path)))[..8] + ".json";
        var jsonPath = Path.Combine(output, name);
        File.WriteAllText(jsonPath, objects.ToString(Formatting.Indented));
        var parents = objects.Where(item => item.Value<string>("Type")?.EndsWith("GeneratedClass") == true)
            .Select(item => item["SuperStruct"]?.Value<string>("ObjectPath")).Where(value => value != null)
            .Select(value => value!.Split('.')[0]).ToArray();
        foreach (var parent in parents) pending.Enqueue(parent);
        records.Add(new { package = path, json_file = jsonPath, parents,
            sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(jsonPath))) });
        Console.WriteLine($"BLUEPRINT_PARSED {path}");
    }
    File.WriteAllText(Path.Combine(output, "blueprint_index.json"), JsonConvert.SerializeObject(records, Formatting.Indented));
    return;
}
if (args[3] == "metadata")
{
    var scope = JObject.Parse(File.ReadAllText(args[4]));
    foreach (var entry in scope["models"]!)
    {
        var ident = entry.Value<string>("ident")!;
        var directory = Path.Combine(output, ident);
        var exportPath = Path.Combine(directory, "export.json");
        if (!File.Exists(exportPath)) continue;
        var exports = JArray.Parse(File.ReadAllText(exportPath));
        if (exports.Any(item => !item.Value<bool>("Success"))) continue;
        var mesh = provider.LoadPackage(entry.Value<string>("package")!).GetExports().OfType<USkinnedAsset>().Single();
        if (!File.Exists(Path.Combine(directory, "texture_metadata.json")))
            MeshDump.Write(mesh, directory);
        var records = new List<object>();
        foreach (var item in exports)
        {
            var objectPath = item.Value<string>("ObjectPath")!;
            var files = item["DiskFilePaths"]!.Values<string>().ToArray();
            if (!files.Any(path => new[] { ".png", ".hdr", ".dds", ".exr" }
                    .Any(extension => path!.EndsWith(extension, StringComparison.OrdinalIgnoreCase)))) continue;
            var texture = provider.LoadPackageObject<UTexture>(objectPath);
            records.Add(new { source = objectPath, texture.SRGB, normal_map = texture.IsNormalMap,
                compression = texture.CompressionSettings.ToString(), format = texture.Format.ToString(), files });
        }
        File.WriteAllText(Path.Combine(directory, "texture_metadata.json"), JsonConvert.SerializeObject(records, Formatting.Indented));
        var packages = exports.Select(item => item.Value<string>("ObjectPath")!.Split('.')[0])
            .Append(entry.Value<string>("package")!.Replace(".uasset", "")).Distinct();
        var sources = new List<object>();
        foreach (var package in packages)
        foreach (var extension in new[] { ".uasset", ".uexp", ".ubulk", ".uptnl" })
        {
            if (!provider.TryGetGameFile(package + extension, out var file)) continue;
            sources.Add(new { file.Path, file.Size, archive = (file as VfsEntry)?.Vfs.Path,
                sha256 = Convert.ToHexStringLower(SHA256.HashData(file.Read())) });
        }
        File.WriteAllText(Path.Combine(directory, "source_provenance.json"), JsonConvert.SerializeObject(sources, Formatting.Indented));
        Console.WriteLine($"METADATA_COMPLETED {ident} textures={records.Count}");
        GC.Collect();
    }
    return;
}
if (args[3] == "anim")
{
    // each AnimSequence package listed in args[4] as its skeleton bones' local transforms on every frame, sampled
    // by CUE4Parse's own decoder (ConvertAnims, CAnimTrack.GetBoneTransform), in Unreal's space and units:
    // <k>.json (the sequence, its skeleton, rate, frames, bone names, which bones have keys) and <k>.bin
    // (float32 frames × bones × [qx qy qz qw px py pz sx sy sz]); a bone without keys holds the skeleton's ref pose
    var listed = File.ReadAllLines(args[4]).Where(line => line.Trim().Length > 0).ToArray();
    for (var k = 0; k < listed.Length; k++)
    {
        var package = listed[k].Trim();
        var sequence = provider.LoadPackage(package).GetExports().OfType<UAnimSequence>().FirstOrDefault();
        if (sequence == null) { Console.WriteLine($"ANIM_SKIPPED {package} no AnimSequence"); continue; }
        var skeleton = sequence.Skeleton.Load<USkeleton>();
        if (skeleton == null) { Console.WriteLine($"ANIM_SKIPPED {package} no skeleton"); continue; }
        var reference = skeleton.ReferenceSkeleton;
        var count = reference.FinalRefBoneInfo.Length;
        if (sequence.CompressedDataStructure == null && sequence.SerializedBoneByteStream is { Length: > 0 } stream)
        {
            // a codec this parser cannot decode (the game's ACL one): its buffer as serialized, which track drives
            // which skeleton bone, and the skeleton's ref pose for the bones no track drives; decoded by the caller
            File.WriteAllBytes(Path.Combine(output, $"{k}.acl"), stream);
            File.WriteAllText(Path.Combine(output, $"{k}.json"), JsonConvert.SerializeObject(new
            {
                package, name = sequence.Name, skeleton = skeleton.GetPathName(), codec = sequence.BoneCodecDDCHandle,
                frames = sequence.NumFrames, length = sequence.SequenceLength, rate_scale = sequence.RateScale,
                additive = sequence.AdditiveAnimType != EAdditiveAnimationType.AAT_None,
                bones = reference.FinalRefBoneInfo.Select(info => info.Name.Text),
                tracks = sequence.CompressedTrackToSkeletonMapTable.Select(map => map.BoneTreeIndex),
                reference = reference.FinalRefBonePose.Select(rest => new[] { rest.Rotation.X, rest.Rotation.Y,
                    rest.Rotation.Z, rest.Rotation.W, rest.Translation.X, rest.Translation.Y, rest.Translation.Z,
                    rest.Scale3D.X, rest.Scale3D.Y, rest.Scale3D.Z })
            }, Formatting.Indented));
            Console.WriteLine($"ANIM_RAW {package} codec={sequence.BoneCodecDDCHandle} bytes={stream.Length}");
            continue;
        }
        var converted = skeleton.ConvertAnims(sequence).Sequences[0];
        var frames = Math.Max(converted.NumFrames, 1);
        using (var bin = new BinaryWriter(File.Create(Path.Combine(output, $"{k}.bin"))))
            for (var frame = 0; frame < frames; frame++)
            for (var bone = 0; bone < count; bone++)
            {
                var rest = reference.FinalRefBonePose[bone];
                var quat = rest.Rotation;
                var position = rest.Translation;
                var scale = rest.Scale3D;
                if (bone < converted.Tracks.Count)
                    converted.Tracks[bone].GetBoneTransform(frame, frames, ref quat, ref position, ref scale);
                foreach (var value in new[] { quat.X, quat.Y, quat.Z, quat.W, position.X, position.Y, position.Z,
                             scale.X, scale.Y, scale.Z })
                    bin.Write(value);
            }
        File.WriteAllText(Path.Combine(output, $"{k}.json"), JsonConvert.SerializeObject(new
        {
            package, name = sequence.Name, skeleton = skeleton.GetPathName(), rate = converted.FramesPerSecond,
            frames, additive = converted.IsAdditive,
            bones = reference.FinalRefBoneInfo.Select(info => info.Name.Text),
            keyed = Enumerable.Range(0, count).Select(bone => bone < converted.Tracks.Count && converted.Tracks[bone].HasKeys())
        }, Formatting.Indented));
        Console.WriteLine($"ANIM_WRITTEN {package} frames={frames} bones={count}");
    }
    return;
}
if (args[3] == "mesh")
{
    var objects = provider.LoadPackage(args[4]).GetExports().ToArray();
    var meshes = objects.OfType<USkinnedAsset>().ToArray();
    if (meshes.Length == 0) throw new InvalidDataException("No skeletal mesh in package");
    var session = new ExportSession { MaxDegreeOfParallelism = 1 };
    foreach (var mesh in meshes)
    {
        MeshDump.Write(mesh, output);
        session.Add(mesh);
    }
    var results = await session.RunAsync(output, new ExportOptions(materialDepth: EMaterialDepth.AllLayers));
    File.WriteAllText(Path.Combine(output, "export.json"), JsonConvert.SerializeObject(results, Formatting.Indented));
    if (results.Any(result => !result.Success)) throw new InvalidDataException("Export contains failed dependencies");
    return;
}
throw new ArgumentException("Unknown mode");
