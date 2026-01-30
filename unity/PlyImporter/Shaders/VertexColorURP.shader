// Vertex Color shader for Universal Render Pipeline (URP)
// Use this if your project uses URP instead of Built-in Render Pipeline
// NOTE: This shader requires URP to be installed. If you're using Built-in RP,
// use Custom/VertexColor or Custom/VertexColorUnlit instead.

Shader "Custom/VertexColorURP"
{
    Properties
    {
        [Enum(Off,0,Front,1,Back,2)] _Cull ("Cull Mode", Float) = 2
        [Toggle] _FlipNormals ("Flip Normals", Float) = 0
    }

    SubShader
    {
        Tags 
        { 
            "RenderType" = "Opaque" 
            "RenderPipeline" = "UniversalPipeline"
            "Queue" = "Geometry"
        }

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }

            Cull [_Cull]

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_fog
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS_CASCADE

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            float _FlipNormals;

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                float4 color : COLOR;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float4 color : COLOR;
                float3 normalWS : TEXCOORD0;
                float3 positionWS : TEXCOORD1;
                float fogFactor : TEXCOORD2;
            };

            Varyings vert(Attributes input)
            {
                Varyings output;

                // Apply normal flip if enabled
                float3 normalOS = input.normalOS;
                if (_FlipNormals > 0.5)
                {
                    normalOS = -normalOS;
                }

                VertexPositionInputs posInputs = GetVertexPositionInputs(input.positionOS.xyz);
                VertexNormalInputs normInputs = GetVertexNormalInputs(normalOS);

                output.positionCS = posInputs.positionCS;
                output.positionWS = posInputs.positionWS;
                output.normalWS = normInputs.normalWS;
                output.color = input.color;
                output.fogFactor = ComputeFogFactor(posInputs.positionCS.z);

                return output;
            }

            half4 frag(Varyings input, bool isFrontFace : SV_IsFrontFace) : SV_Target
            {
                // Get main light
                Light mainLight = GetMainLight();

                // Calculate simple diffuse lighting
                // Flip normal for back faces when rendering double-sided
                float3 normalWS = normalize(input.normalWS);
                if (!isFrontFace)
                {
                    normalWS = -normalWS;
                }
                float NdotL = saturate(dot(normalWS, mainLight.direction)) * 0.5 + 0.5;

                // Apply vertex color with lighting
                half4 color = input.color;
                color.rgb *= NdotL * mainLight.color;

                // Apply fog
                color.rgb = MixFog(color.rgb, input.fogFactor);

                return color;
            }
            ENDHLSL
        }

        // Shadow caster pass - compatible with all URP versions
        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }

            ZWrite On
            ZTest LEqual
            ColorMask 0
            Cull [_Cull]

            HLSLPROGRAM
            #pragma vertex ShadowPassVertex
            #pragma fragment ShadowPassFragment

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
            };

            float3 _LightDirection;
            float4 _ShadowBias; // x: depth bias, y: normal bias

            // Custom shadow bias function to avoid dependency on deprecated URP functions
            float4 GetShadowPositionHClip(Attributes input)
            {
                float3 positionWS = TransformObjectToWorld(input.positionOS.xyz);
                float3 normalWS = TransformObjectToWorldNormal(input.normalOS);

                // Apply shadow bias
                float invNdotL = 1.0 - saturate(dot(_LightDirection, normalWS));
                float scale = invNdotL * _ShadowBias.y;

                // Normal bias
                positionWS = positionWS + normalWS * scale.xxx;

                float4 positionCS = TransformWorldToHClip(positionWS);

                // Depth bias
                #if UNITY_REVERSED_Z
                    positionCS.z += _ShadowBias.x;
                    positionCS.z = min(positionCS.z, positionCS.w * 0.99999f);
                #else
                    positionCS.z -= _ShadowBias.x;
                    positionCS.z = max(positionCS.z, positionCS.w * 0.00001f);
                #endif

                return positionCS;
            }

            Varyings ShadowPassVertex(Attributes input)
            {
                Varyings output;
                output.positionCS = GetShadowPositionHClip(input);
                return output;
            }

            half4 ShadowPassFragment(Varyings input) : SV_TARGET
            {
                return 0;
            }
            ENDHLSL
        }

        // Depth only pass for depth prepass
        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode" = "DepthOnly" }

            ZWrite On
            ColorMask 0
            Cull [_Cull]

            HLSLPROGRAM
            #pragma vertex DepthOnlyVertex
            #pragma fragment DepthOnlyFragment

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
            };

            Varyings DepthOnlyVertex(Attributes input)
            {
                Varyings output;
                output.positionCS = TransformObjectToHClip(input.positionOS.xyz);
                return output;
            }

            half4 DepthOnlyFragment(Varyings input) : SV_TARGET
            {
                return 0;
            }
            ENDHLSL
        }
    }

    // Fallback to built-in vertex color shader
    FallBack "Custom/VertexColor"
}
