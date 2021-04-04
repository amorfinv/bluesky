#version 330
 
// Interpolated values from the vertex shaders
in vec2 texcoords_fs;
in vec3 v_color;

// Ouput data
out vec4 color;
 
// Values that stay constant for the whole mesh.
uniform sampler2DArray tex_sampler;
 
void main()
{ 
    // Output color = color of the texture at the specified UV
    color = texture(tex_sampler, vec3(texcoords_fs, 0.0));
}
