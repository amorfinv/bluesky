#version 330
#define VERTEX_IS_LATLON 0

// Uniform block of global data
layout (std140) uniform global_data {
int wrap_dir;           // Wrap-around direction
float wrap_lon;         // Wrap-around longitude
float panlat;           // Map panning coordinates [deg]
float panlon;           // Map panning coordinates [deg]
float zoom;             // Screen zoom factor [-]
int screen_width;       // Screen width in pixels
int screen_height;      // Screen height in pixels
};

layout (location = 0) in vec2 vertex_in;
layout (location = 1) in vec2 texcoords_in;
layout (location = 2) in vec3 a_color;

out vec2 texcoords_fs;
out vec3 v_color;

void main()
{

	vec2 vAR = vec2(1.0, float(screen_width) / float(screen_height));
	vec2 flat_earth = vec2(cos(radians(panlat)), 1.0);

	// Vertex position and rotation calculations
	vec2 position = vec2(0, 0);
	position -= vec2(panlon, panlat);

	// Lat/lon vertex coordinates are flipped: lat is index 0, but screen y-axis, and lon is index 1, but screen x-axis
	gl_Position = vec4(vAR * flat_earth * zoom * (position + vertex_in.yx), 0.0, 1.0);
	texcoords_fs = texcoords_in.ts;

	// color to fragment shader
	v_color = a_color;
}