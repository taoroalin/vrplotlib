import * as twgl from "twgl.js";
let programInfo, bufferInfo;
export function copyTexture(gl, input, output, width, height, rgb = false) {
  const format = rgb ? gl.RGBA32F : gl.R32F;
  // const format = gl.R32F;
  const stime = performance.now()
  const fbi = twgl.createFramebufferInfo(gl, [{ attachment: input, format: format, type: gl.FLOAT }], width, height);
  twgl.bindFramebufferInfo(gl, fbi);
  gl.bindTexture(gl.TEXTURE_2D, output)
  // HACK HERE: only copying top corner because that "magically" works with tensorflow internal format
  if (rgb) {
    gl.copyTexImage2D(gl.TEXTURE_2D, 0, format, 0, 0, width, height, 0);

  } else {

    gl.copyTexImage2D(gl.TEXTURE_2D, 0, format, 0, 0, width * 2, height * 2, 0);
  }
  // gl.generateMipmap(gl.TEXTURE_2D)
  gl.bindFramebuffer(gl.FRAMEBUFFER, null)
  // console.log(`copytexture took ${performance.now() - stime}`)
}

export function withAsFramebuffer(gl, texture, width, height, f) {
  const fbi = twgl.createFramebufferInfo(gl, [{ attachment: texture, format: gl.RGBA, type: gl.FLOAT }], width, height)
  twgl.bindFramebufferInfo(gl, fbi)
  f()
  gl.bindFramebuffer(gl.FRAMEBUFFER, null)
}

export function copyTextureWah(gl, input, output, width, height) {
  const stime = performance.now()
  const format = gl.RGBA32F;
  console.log(input, output)
  gl.bindTexture(gl.TEXTURE_2D, output)
  const fbi = twgl.createFramebufferInfo(gl, [{ attachment: output, format: format, type: gl.FLOAT }], width, height);
  console.log(fbi)
  twgl.bindFramebufferInfo(gl, fbi);
  const vs = `attribute vec2 a_position;

uniform mat3 u_matrix;

varying vec2 v_texCoord;

void main() {
   gl_Position = vec4(u_matrix * vec3(a_position, 1), 1);

   // because we're using a unit quad we can just use
   // the same data for our texcoords.
   v_texCoord = a_position;  
}`
  const fs =
    `precision mediump float;

// our texture
uniform sampler2D u_image;

// the texCoords passed in from the vertex shader.
varying vec2 v_texCoord;

void main() {
   gl_FragColor = texture2D(u_image, v_texCoord);
}
  `
  const programInfo = twgl.createProgramInfo(gl, [fs, vs])

  const bufferInfo = twgl.createBufferInfoFromArrays(gl, {
    position: {
      numComponents: 3, data: [
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        1, 1, 0],
    },
    texcoord: { numComponents: 2, data: [0, 0, 0, 1, 1, 0, 1, 1], },
  })
  twgl.setBuffersAndAttributes(gl, programInfo, bufferInfo)
  gl.bindTexture(gl.TEXTURE_2D, input)
  const uniforms = {
    inputTexture: input,
    outputResolution: [width, height],
  }
  twgl.setUniforms(programInfo, uniforms)

  gl.drawBufferInfo(gl, bufferInfo)

  gl.bindTexture(gl.TEXTURE_2D, output)
  gl.generateMipmap(gl.TEXTURE_2D)

  gl.bindFramebuffer(gl.FRAMEBUFFER, null)
  // undo gl state changes
  // gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}
  // const buf = gl.createBuffer();
  // gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  // gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
  //   -1, -1,
  //   1, -1,
  //   -1, 1,
  //   -1, 1,
  //   1, -1,
  //   1, 1,
  // ]), gl.STATIC_DRAW);

  // function setAttributes(buf, positionLoc) {
  //   gl.enableVertexAttribArray(positionLoc);
  //   gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
  // }

  // const colorPrgPositionLoc = gl.getAttribLocation(colorProgram, "position");
  // setAttributes(buf, colorPrgPositionLoc);
  // const colorLoc = gl.getUniformLocation(colorProgram, "color");
  // const colorProgMatrixLoc = gl.getUniformLocation(colorProgram, "matrix");

  // // draw red rect to first texture through the framebuffer it's attached to
  // gl.useProgram(colorProgram);

  // gl.bindFramebuffer(gl.FRAMEBUFFER, texFbPair1.fb);
  // gl.viewport(0, 0, 64, 64);
  // gl.uniform4fv(colorLoc, [1, 0, 0, 1]);
  // gl.uniformMatrix4fv(colorProgMatrixLoc, false, [
  //   0.5, 0, 0, 0,
  //   0, .25, 0, 0,
  //   0, 0, 1, 0,
  //   .2, .3, 0, 1,
  // ]);

  // gl.drawArrays(gl.TRIANGLES, 0, 6);

  // // Draw a blue rect to the second texture through the framebuffer it's attached to
  // gl.bindFramebuffer(gl.FRAMEBUFFER, texFbPair2.fb);
  // gl.viewport(0, 0, 64, 64);
  // gl.uniform4fv(colorLoc, [0, 0, 1, 1]);
  // gl.uniformMatrix4fv(colorProgMatrixLoc, false, [
  //   0.25, 0, 0, 0,
  //   0, .5, 0, 0,
  //   0, 0, 1, 0,
  //   .2, .3, 0, 1,
  // ]);

  // gl.drawArrays(gl.TRIANGLES, 0, 6);

  // // Draw both textures to the canvas
  // gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  // gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);



  // const mixPrgPositionLoc = gl.getAttribLocation(copyProgram, "position");
  // setAttributes(buf, mixPrgPositionLoc);

  // const mixProgMatrixLoc = gl.getUniformLocation(copyProgram, "matrix");

  // const tex1Loc = gl.getUniformLocation(copyProgram, "tex1");
  // const tex2Loc = gl.getUniformLocation(copyProgram, "tex2");

  // gl.useProgram(copyProgram);

  // gl.uniform1i(tex1Loc, 0);
  // gl.uniform1i(tex2Loc, 1);
  // gl.activeTexture(gl.TEXTURE0 + 0);
  // gl.bindTexture(gl.TEXTURE_2D, texFbPair1.tex);
  // gl.activeTexture(gl.TEXTURE0 + 1);
  // gl.bindTexture(gl.TEXTURE_2D, texFbPair2.tex);
  // gl.uniformMatrix4fv(mixProgMatrixLoc, false, [
  //   1, 0, 0, 0,
  //   0, 1, 0, 0,
  //   0, 0, 1, 0,
  //   0, 0, 0, 1,
  // ]);

  // gl.drawArrays(gl.TRIANGLES, 0, 6);