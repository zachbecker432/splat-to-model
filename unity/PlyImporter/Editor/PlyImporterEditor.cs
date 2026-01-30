#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEditor.AssetImporters;
using System.IO;

/// <summary>
/// Custom asset importer for PLY files in Unity Editor.
/// Automatically imports PLY files with vertex colors when dropped into the Assets folder.
/// </summary>
[ScriptedImporter(1, "ply")]
public class PlyImporterEditor : ScriptedImporter
{
    [Tooltip("Swap Y and Z axes (useful for converting coordinate systems)")]
    public bool swapYZ = false;

    [Tooltip("Scale factor to apply to the mesh")]
    public float scale = 1f;

    [Tooltip("Recalculate normals even if PLY has normals")]
    public bool recalculateNormals = false;

    public override void OnImportAsset(AssetImportContext ctx)
    {
        // Import the PLY file
        var mesh = PlyImporter.Import(ctx.assetPath, swapYZ, scale);
        
        if (mesh == null)
        {
            ctx.LogImportError("Failed to import PLY file");
            return;
        }

        if (recalculateNormals)
        {
            mesh.RecalculateNormals();
        }

        // Create a GameObject with the mesh
        var go = new GameObject(Path.GetFileNameWithoutExtension(ctx.assetPath));
        var meshFilter = go.AddComponent<MeshFilter>();
        var meshRenderer = go.AddComponent<MeshRenderer>();

        meshFilter.sharedMesh = mesh;

        // Create and assign a default vertex color material
        var material = CreateVertexColorMaterial();
        meshRenderer.sharedMaterial = material;

        // Add assets to the import context
        ctx.AddObjectToAsset("mesh", mesh);
        ctx.AddObjectToAsset("material", material);
        ctx.AddObjectToAsset("prefab", go);
        ctx.SetMainObject(go);
    }

    private Material CreateVertexColorMaterial()
    {
        Shader shader = null;
        
        // Detect render pipeline and choose appropriate shader
        if (IsUsingURP())
        {
            // Try URP vertex color shader first
            shader = Shader.Find("Custom/VertexColorURP");
            
            if (shader == null)
            {
                // Fall back to URP Lit shader
                shader = Shader.Find("Universal Render Pipeline/Lit");
            }
            
            if (shader == null)
            {
                // Try URP Simple Lit
                shader = Shader.Find("Universal Render Pipeline/Simple Lit");
            }
        }
        else if (IsUsingHDRP())
        {
            // HDRP fallback
            shader = Shader.Find("HDRP/Lit");
        }
        
        // If no pipeline-specific shader found, try custom Built-in shaders
        if (shader == null)
        {
            shader = Shader.Find("Custom/VertexColor");
        }
        
        if (shader == null)
        {
            shader = Shader.Find("Custom/VertexColorUnlit");
        }
        
        if (shader == null)
        {
            // Fall back to a built-in shader that shows vertex colors
            shader = Shader.Find("Particles/Standard Unlit");
        }

        if (shader == null)
        {
            // Last resort - use standard shader
            shader = Shader.Find("Standard");
        }

        var material = new Material(shader);
        material.name = "VertexColorMaterial";

        // Configure material based on shader type
        if (shader.name == "Particles/Standard Unlit")
        {
            material.SetFloat("_ColorMode", 1); // Vertex colors
        }
        else if (shader.name == "Custom/VertexColorURP")
        {
            // Set default cull mode to Back (normal)
            material.SetFloat("_Cull", 2);
            material.SetFloat("_FlipNormals", 0);
        }

        string pipelineName = IsUsingURP() ? "URP" : (IsUsingHDRP() ? "HDRP" : "Built-in");
        Debug.Log($"PlyImporter: Created material with shader '{shader.name}' for {pipelineName} pipeline");

        return material;
    }
    
    /// <summary>
    /// Check if the project is using Universal Render Pipeline (URP)
    /// </summary>
    private static bool IsUsingURP()
    {
        var currentRP = UnityEngine.Rendering.GraphicsSettings.currentRenderPipeline;
        if (currentRP == null)
            return false;
            
        // Check if it's URP by type name (works without direct reference to URP assembly)
        string typeName = currentRP.GetType().Name;
        return typeName.Contains("Universal") || typeName.Contains("URP") || typeName.Contains("Lightweight");
    }
    
    /// <summary>
    /// Check if the project is using High Definition Render Pipeline (HDRP)
    /// </summary>
    private static bool IsUsingHDRP()
    {
        var currentRP = UnityEngine.Rendering.GraphicsSettings.currentRenderPipeline;
        if (currentRP == null)
            return false;
            
        string typeName = currentRP.GetType().Name;
        return typeName.Contains("HDRenderPipeline") || typeName.Contains("HDRP") || typeName.Contains("HighDefinition");
    }
}

/// <summary>
/// Menu items for manually importing PLY files
/// </summary>
public static class PlyImporterMenu
{
    [MenuItem("Assets/Import PLY File", false, 30)]
    public static void ImportPlyFile()
    {
        string path = EditorUtility.OpenFilePanel("Select PLY File", "", "ply");
        if (string.IsNullOrEmpty(path))
            return;

        ImportPlyAtPath(path, false, 1f);
    }

    [MenuItem("Assets/Import PLY File (Swap YZ)", false, 31)]
    public static void ImportPlyFileSwapYZ()
    {
        string path = EditorUtility.OpenFilePanel("Select PLY File", "", "ply");
        if (string.IsNullOrEmpty(path))
            return;

        ImportPlyAtPath(path, true, 1f);
    }

    private static void ImportPlyAtPath(string sourcePath, bool swapYZ, float scale)
    {
        var mesh = PlyImporter.Import(sourcePath, swapYZ, scale);
        if (mesh == null)
        {
            EditorUtility.DisplayDialog("Import Failed", "Failed to import PLY file.", "OK");
            return;
        }

        // Save mesh as asset
        string fileName = Path.GetFileNameWithoutExtension(sourcePath);
        string assetPath = $"Assets/{fileName}.asset";
        assetPath = AssetDatabase.GenerateUniqueAssetPath(assetPath);

        AssetDatabase.CreateAsset(mesh, assetPath);

        // Create prefab with mesh and vertex color material
        var go = new GameObject(fileName);
        var meshFilter = go.AddComponent<MeshFilter>();
        var meshRenderer = go.AddComponent<MeshRenderer>();

        meshFilter.sharedMesh = mesh;

        // Create material with proper shader for current render pipeline
        var material = CreateVertexColorMaterialStatic();
        material.name = $"{fileName}_Material";

        string materialPath = $"Assets/{fileName}_Material.mat";
        materialPath = AssetDatabase.GenerateUniqueAssetPath(materialPath);
        AssetDatabase.CreateAsset(material, materialPath);

        meshRenderer.sharedMaterial = material;

        string prefabPath = $"Assets/{fileName}.prefab";
        prefabPath = AssetDatabase.GenerateUniqueAssetPath(prefabPath);
        PrefabUtility.SaveAsPrefabAsset(go, prefabPath);

        Object.DestroyImmediate(go);

        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();

        // Select the created prefab
        var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
        Selection.activeObject = prefab;
        EditorGUIUtility.PingObject(prefab);

        Debug.Log($"PLY imported successfully: {assetPath}");
    }
    
    /// <summary>
    /// Static version of CreateVertexColorMaterial for use in menu commands
    /// </summary>
    private static Material CreateVertexColorMaterialStatic()
    {
        Shader shader = null;
        
        // Detect render pipeline and choose appropriate shader
        if (IsUsingURPStatic())
        {
            shader = Shader.Find("Custom/VertexColorURP");
            if (shader == null)
                shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null)
                shader = Shader.Find("Universal Render Pipeline/Simple Lit");
        }
        else if (IsUsingHDRPStatic())
        {
            shader = Shader.Find("HDRP/Lit");
        }
        
        if (shader == null)
            shader = Shader.Find("Custom/VertexColor");
        if (shader == null)
            shader = Shader.Find("Custom/VertexColorUnlit");
        if (shader == null)
            shader = Shader.Find("Particles/Standard Unlit");
        if (shader == null)
            shader = Shader.Find("Standard");

        var material = new Material(shader);

        if (shader.name == "Particles/Standard Unlit")
        {
            material.SetFloat("_ColorMode", 1);
        }
        else if (shader.name == "Custom/VertexColorURP")
        {
            material.SetFloat("_Cull", 2);
            material.SetFloat("_FlipNormals", 0);
        }

        return material;
    }
    
    private static bool IsUsingURPStatic()
    {
        var currentRP = UnityEngine.Rendering.GraphicsSettings.currentRenderPipeline;
        if (currentRP == null) return false;
        string typeName = currentRP.GetType().Name;
        return typeName.Contains("Universal") || typeName.Contains("URP") || typeName.Contains("Lightweight");
    }
    
    private static bool IsUsingHDRPStatic()
    {
        var currentRP = UnityEngine.Rendering.GraphicsSettings.currentRenderPipeline;
        if (currentRP == null) return false;
        string typeName = currentRP.GetType().Name;
        return typeName.Contains("HDRenderPipeline") || typeName.Contains("HDRP") || typeName.Contains("HighDefinition");
    }
}
#endif
