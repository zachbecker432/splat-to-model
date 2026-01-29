using UnityEngine;
using System;
using System.IO;
using System.Collections.Generic;
using System.Globalization;

/// <summary>
/// PLY file importer for Unity with vertex color support.
/// Supports both ASCII and binary (little-endian) PLY formats.
/// 
/// Usage:
///   var mesh = PlyImporter.Import("path/to/file.ply");
///   GetComponent<MeshFilter>().mesh = mesh;
/// </summary>
public static class PlyImporter
{
    private enum PlyFormat
    {
        Ascii,
        BinaryLittleEndian,
        BinaryBigEndian
    }

    private class PlyProperty
    {
        public string Name;
        public string Type;
        public bool IsList;
        public string ListCountType;
        public string ListValueType;
    }

    private class PlyElement
    {
        public string Name;
        public int Count;
        public List<PlyProperty> Properties = new List<PlyProperty>();
    }

    /// <summary>
    /// Import a PLY file and create a Unity Mesh.
    /// </summary>
    /// <param name="path">Full path to the PLY file</param>
    /// <param name="swapYZ">Swap Y and Z axes (common for converting between coordinate systems)</param>
    /// <param name="scale">Scale factor to apply to vertices</param>
    /// <returns>Unity Mesh with vertices, triangles, normals (if present), and vertex colors (if present)</returns>
    public static Mesh Import(string path, bool swapYZ = false, float scale = 1f)
    {
        if (!File.Exists(path))
        {
            Debug.LogError($"PlyImporter: File not found: {path}");
            return null;
        }

        using (var stream = File.OpenRead(path))
        using (var reader = new BinaryReader(stream))
        {
            // Parse header
            var format = PlyFormat.Ascii;
            var elements = new List<PlyElement>();
            
            string line;
            while ((line = ReadLine(reader)) != null)
            {
                line = line.Trim();
                if (string.IsNullOrEmpty(line) || line.StartsWith("comment"))
                    continue;

                if (line == "end_header")
                    break;

                var parts = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                
                if (parts[0] == "format")
                {
                    if (parts[1] == "ascii")
                        format = PlyFormat.Ascii;
                    else if (parts[1] == "binary_little_endian")
                        format = PlyFormat.BinaryLittleEndian;
                    else if (parts[1] == "binary_big_endian")
                        format = PlyFormat.BinaryBigEndian;
                }
                else if (parts[0] == "element")
                {
                    elements.Add(new PlyElement
                    {
                        Name = parts[1],
                        Count = int.Parse(parts[2])
                    });
                }
                else if (parts[0] == "property")
                {
                    if (elements.Count == 0) continue;
                    
                    var prop = new PlyProperty();
                    if (parts[1] == "list")
                    {
                        prop.IsList = true;
                        prop.ListCountType = parts[2];
                        prop.ListValueType = parts[3];
                        prop.Name = parts[4];
                    }
                    else
                    {
                        prop.IsList = false;
                        prop.Type = parts[1];
                        prop.Name = parts[2];
                    }
                    elements[elements.Count - 1].Properties.Add(prop);
                }
            }

            // Find vertex and face elements
            PlyElement vertexElement = null;
            PlyElement faceElement = null;
            
            foreach (var elem in elements)
            {
                if (elem.Name == "vertex")
                    vertexElement = elem;
                else if (elem.Name == "face")
                    faceElement = elem;
            }

            if (vertexElement == null)
            {
                Debug.LogError("PlyImporter: No vertex element found in PLY file");
                return null;
            }

            // Determine property indices
            int xIdx = -1, yIdx = -1, zIdx = -1;
            int nxIdx = -1, nyIdx = -1, nzIdx = -1;
            int rIdx = -1, gIdx = -1, bIdx = -1, aIdx = -1;

            for (int i = 0; i < vertexElement.Properties.Count; i++)
            {
                var propName = vertexElement.Properties[i].Name.ToLower();
                switch (propName)
                {
                    case "x": xIdx = i; break;
                    case "y": yIdx = i; break;
                    case "z": zIdx = i; break;
                    case "nx": nxIdx = i; break;
                    case "ny": nyIdx = i; break;
                    case "nz": nzIdx = i; break;
                    case "red": case "r": rIdx = i; break;
                    case "green": case "g": gIdx = i; break;
                    case "blue": case "b": bIdx = i; break;
                    case "alpha": case "a": aIdx = i; break;
                }
            }

            bool hasNormals = nxIdx >= 0 && nyIdx >= 0 && nzIdx >= 0;
            bool hasColors = rIdx >= 0 && gIdx >= 0 && bIdx >= 0;

            Debug.Log($"PlyImporter: Loading {vertexElement.Count} vertices, " +
                     $"hasNormals={hasNormals}, hasColors={hasColors}, format={format}");

            // Read vertex data
            var vertices = new Vector3[vertexElement.Count];
            var normals = hasNormals ? new Vector3[vertexElement.Count] : null;
            var colors = hasColors ? new Color32[vertexElement.Count] : null;

            if (format == PlyFormat.Ascii)
            {
                ReadVerticesAscii(reader, vertexElement, vertices, normals, colors,
                    xIdx, yIdx, zIdx, nxIdx, nyIdx, nzIdx, rIdx, gIdx, bIdx, aIdx,
                    swapYZ, scale);
            }
            else
            {
                bool bigEndian = format == PlyFormat.BinaryBigEndian;
                ReadVerticesBinary(reader, vertexElement, vertices, normals, colors,
                    xIdx, yIdx, zIdx, nxIdx, nyIdx, nzIdx, rIdx, gIdx, bIdx, aIdx,
                    swapYZ, scale, bigEndian);
            }

            // Read face data
            int[] triangles = null;
            if (faceElement != null && faceElement.Count > 0)
            {
                if (format == PlyFormat.Ascii)
                {
                    triangles = ReadFacesAscii(reader, faceElement);
                }
                else
                {
                    bool bigEndian = format == PlyFormat.BinaryBigEndian;
                    triangles = ReadFacesBinary(reader, faceElement, bigEndian);
                }
            }

            // Create mesh
            var mesh = new Mesh();
            mesh.name = Path.GetFileNameWithoutExtension(path);

            // Use 32-bit indices if needed
            if (vertices.Length > 65535)
            {
                mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            }

            mesh.vertices = vertices;
            
            if (hasNormals)
                mesh.normals = normals;
            
            if (hasColors)
                mesh.colors32 = colors;

            if (triangles != null && triangles.Length > 0)
            {
                mesh.triangles = triangles;
            }

            // Recalculate normals if not present
            if (!hasNormals && triangles != null)
            {
                mesh.RecalculateNormals();
            }

            mesh.RecalculateBounds();

            Debug.Log($"PlyImporter: Created mesh with {vertices.Length} vertices, " +
                     $"{(triangles?.Length ?? 0) / 3} triangles, hasColors={hasColors}");

            return mesh;
        }
    }

    private static string ReadLine(BinaryReader reader)
    {
        var chars = new List<char>();
        try
        {
            while (true)
            {
                char c = (char)reader.ReadByte();
                if (c == '\n')
                    break;
                if (c != '\r')
                    chars.Add(c);
            }
            return new string(chars.ToArray());
        }
        catch (EndOfStreamException)
        {
            return chars.Count > 0 ? new string(chars.ToArray()) : null;
        }
    }

    private static void ReadVerticesAscii(BinaryReader reader, PlyElement element,
        Vector3[] vertices, Vector3[] normals, Color32[] colors,
        int xIdx, int yIdx, int zIdx, int nxIdx, int nyIdx, int nzIdx,
        int rIdx, int gIdx, int bIdx, int aIdx, bool swapYZ, float scale)
    {
        var culture = CultureInfo.InvariantCulture;
        
        for (int i = 0; i < element.Count; i++)
        {
            string line = ReadLine(reader);
            if (string.IsNullOrEmpty(line))
            {
                i--;
                continue;
            }

            var parts = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);

            float x = float.Parse(parts[xIdx], culture) * scale;
            float y = float.Parse(parts[yIdx], culture) * scale;
            float z = float.Parse(parts[zIdx], culture) * scale;

            if (swapYZ)
                vertices[i] = new Vector3(x, z, y);
            else
                vertices[i] = new Vector3(x, y, z);

            if (normals != null)
            {
                float nx = float.Parse(parts[nxIdx], culture);
                float ny = float.Parse(parts[nyIdx], culture);
                float nz = float.Parse(parts[nzIdx], culture);

                if (swapYZ)
                    normals[i] = new Vector3(nx, nz, ny);
                else
                    normals[i] = new Vector3(nx, ny, nz);
            }

            if (colors != null)
            {
                // Determine if colors are 0-255 or 0-1 range
                float r = float.Parse(parts[rIdx], culture);
                float g = float.Parse(parts[gIdx], culture);
                float b = float.Parse(parts[bIdx], culture);
                float a = aIdx >= 0 ? float.Parse(parts[aIdx], culture) : 255f;

                // If values are <= 1, assume 0-1 range
                if (r <= 1f && g <= 1f && b <= 1f)
                {
                    colors[i] = new Color32(
                        (byte)(r * 255),
                        (byte)(g * 255),
                        (byte)(b * 255),
                        (byte)(a <= 1f ? a * 255 : a)
                    );
                }
                else
                {
                    colors[i] = new Color32(
                        (byte)r,
                        (byte)g,
                        (byte)b,
                        (byte)a
                    );
                }
            }
        }
    }

    private static void ReadVerticesBinary(BinaryReader reader, PlyElement element,
        Vector3[] vertices, Vector3[] normals, Color32[] colors,
        int xIdx, int yIdx, int zIdx, int nxIdx, int nyIdx, int nzIdx,
        int rIdx, int gIdx, int bIdx, int aIdx, bool swapYZ, float scale, bool bigEndian)
    {
        for (int i = 0; i < element.Count; i++)
        {
            var values = new object[element.Properties.Count];
            
            for (int p = 0; p < element.Properties.Count; p++)
            {
                values[p] = ReadBinaryValue(reader, element.Properties[p].Type, bigEndian);
            }

            float x = Convert.ToSingle(values[xIdx]) * scale;
            float y = Convert.ToSingle(values[yIdx]) * scale;
            float z = Convert.ToSingle(values[zIdx]) * scale;

            if (swapYZ)
                vertices[i] = new Vector3(x, z, y);
            else
                vertices[i] = new Vector3(x, y, z);

            if (normals != null)
            {
                float nx = Convert.ToSingle(values[nxIdx]);
                float ny = Convert.ToSingle(values[nyIdx]);
                float nz = Convert.ToSingle(values[nzIdx]);

                if (swapYZ)
                    normals[i] = new Vector3(nx, nz, ny);
                else
                    normals[i] = new Vector3(nx, ny, nz);
            }

            if (colors != null)
            {
                float r = Convert.ToSingle(values[rIdx]);
                float g = Convert.ToSingle(values[gIdx]);
                float b = Convert.ToSingle(values[bIdx]);
                float a = aIdx >= 0 ? Convert.ToSingle(values[aIdx]) : 255f;

                // If values are <= 1, assume 0-1 range
                if (r <= 1f && g <= 1f && b <= 1f)
                {
                    colors[i] = new Color32(
                        (byte)(r * 255),
                        (byte)(g * 255),
                        (byte)(b * 255),
                        (byte)(a <= 1f ? a * 255 : a)
                    );
                }
                else
                {
                    colors[i] = new Color32(
                        (byte)r,
                        (byte)g,
                        (byte)b,
                        (byte)a
                    );
                }
            }
        }
    }

    private static int[] ReadFacesAscii(BinaryReader reader, PlyElement element)
    {
        var triangles = new List<int>();
        
        for (int i = 0; i < element.Count; i++)
        {
            string line = ReadLine(reader);
            if (string.IsNullOrEmpty(line))
            {
                i--;
                continue;
            }

            var parts = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            int vertexCount = int.Parse(parts[0]);

            if (vertexCount == 3)
            {
                // Triangle
                triangles.Add(int.Parse(parts[1]));
                triangles.Add(int.Parse(parts[2]));
                triangles.Add(int.Parse(parts[3]));
            }
            else if (vertexCount == 4)
            {
                // Quad - split into two triangles
                int v0 = int.Parse(parts[1]);
                int v1 = int.Parse(parts[2]);
                int v2 = int.Parse(parts[3]);
                int v3 = int.Parse(parts[4]);

                triangles.Add(v0);
                triangles.Add(v1);
                triangles.Add(v2);

                triangles.Add(v0);
                triangles.Add(v2);
                triangles.Add(v3);
            }
            else if (vertexCount > 4)
            {
                // N-gon - fan triangulation
                int v0 = int.Parse(parts[1]);
                for (int j = 2; j < vertexCount; j++)
                {
                    triangles.Add(v0);
                    triangles.Add(int.Parse(parts[j]));
                    triangles.Add(int.Parse(parts[j + 1]));
                }
            }
        }

        return triangles.ToArray();
    }

    private static int[] ReadFacesBinary(BinaryReader reader, PlyElement element, bool bigEndian)
    {
        var triangles = new List<int>();
        var faceProperty = element.Properties[0]; // Assume first property is the vertex_indices list

        for (int i = 0; i < element.Count; i++)
        {
            int vertexCount = Convert.ToInt32(ReadBinaryValue(reader, faceProperty.ListCountType, bigEndian));
            var indices = new int[vertexCount];
            
            for (int j = 0; j < vertexCount; j++)
            {
                indices[j] = Convert.ToInt32(ReadBinaryValue(reader, faceProperty.ListValueType, bigEndian));
            }

            if (vertexCount == 3)
            {
                triangles.Add(indices[0]);
                triangles.Add(indices[1]);
                triangles.Add(indices[2]);
            }
            else if (vertexCount == 4)
            {
                triangles.Add(indices[0]);
                triangles.Add(indices[1]);
                triangles.Add(indices[2]);

                triangles.Add(indices[0]);
                triangles.Add(indices[2]);
                triangles.Add(indices[3]);
            }
            else if (vertexCount > 4)
            {
                for (int j = 1; j < vertexCount - 1; j++)
                {
                    triangles.Add(indices[0]);
                    triangles.Add(indices[j]);
                    triangles.Add(indices[j + 1]);
                }
            }
        }

        return triangles.ToArray();
    }

    private static object ReadBinaryValue(BinaryReader reader, string type, bool bigEndian)
    {
        switch (type)
        {
            case "char":
            case "int8":
                return reader.ReadSByte();
            case "uchar":
            case "uint8":
                return reader.ReadByte();
            case "short":
            case "int16":
                return bigEndian ? ReadInt16BE(reader) : reader.ReadInt16();
            case "ushort":
            case "uint16":
                return bigEndian ? ReadUInt16BE(reader) : reader.ReadUInt16();
            case "int":
            case "int32":
                return bigEndian ? ReadInt32BE(reader) : reader.ReadInt32();
            case "uint":
            case "uint32":
                return bigEndian ? ReadUInt32BE(reader) : reader.ReadUInt32();
            case "float":
            case "float32":
                if (bigEndian)
                {
                    var bytes = reader.ReadBytes(4);
                    Array.Reverse(bytes);
                    return BitConverter.ToSingle(bytes, 0);
                }
                return reader.ReadSingle();
            case "double":
            case "float64":
                if (bigEndian)
                {
                    var bytes = reader.ReadBytes(8);
                    Array.Reverse(bytes);
                    return BitConverter.ToDouble(bytes, 0);
                }
                return reader.ReadDouble();
            default:
                throw new NotSupportedException($"Unsupported PLY type: {type}");
        }
    }

    private static short ReadInt16BE(BinaryReader reader)
    {
        var bytes = reader.ReadBytes(2);
        Array.Reverse(bytes);
        return BitConverter.ToInt16(bytes, 0);
    }

    private static ushort ReadUInt16BE(BinaryReader reader)
    {
        var bytes = reader.ReadBytes(2);
        Array.Reverse(bytes);
        return BitConverter.ToUInt16(bytes, 0);
    }

    private static int ReadInt32BE(BinaryReader reader)
    {
        var bytes = reader.ReadBytes(4);
        Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }

    private static uint ReadUInt32BE(BinaryReader reader)
    {
        var bytes = reader.ReadBytes(4);
        Array.Reverse(bytes);
        return BitConverter.ToUInt32(bytes, 0);
    }
}
