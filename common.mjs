import * as THREE from 'three';
import * as tf from "@tensorflow/tfjs";
import { copyTexture, withAsFramebuffer } from "./gl.mjs";

// import { getDenseTexShape } from "@tensorflow/tfjs-backend-webgl/src/tex_util"
// import { bindVertexProgramAttributeStreams } from "./node_moduls/@tensorflow/tfjs-backend-webgl/src/gpgpu_util";
let gl, renderer, whichState, tfGlState, threeGlState;
let tex_util, webgl_util, gpgpu_util;
let backend, gpgpu;
let playModesSafe = true;

export function threeInternalTexture(threeTex) {
  const props = renderer.properties.get(threeTex)
  return props.__webglTexture
}

export function setRendererAndTf(the_renderer) {
  renderer = the_renderer
  gl = the_renderer.getContext()
  backend = tf.backend()
  gpgpu = backend.gpgpu
  console.log(gpgpu)
  if (backend.tex_util) {
    console.error("CUSTOM WEBGL BACKEND")
    tex_util = backend.tex_util
    gpgpu_util = backend.gpgpu_util
    webgl_util = backend.webgl_util
  }
}

export function glMode() {
  if (playModesSafe || whichState !== "gl") {
  }
  whichState = "gl"
}

export function commonCopyTexture(from, to, width, height, alpha) {
  copyTexture(gl, from, to, width, height, alpha)
}

export function tfMode() {
  if (playModesSafe || whichState !== 'tf') {
    if (gpgpu.framebuffer) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, gpgpu.framebuffer)
    }
    if (gpgpu.vertexAttrsAreBound) {
      // @GLPROBLEM enable scissor
      gl.scissor(0, 0, gl.canvas.width, gl.canvas.height)
      gl.useProgram(gpgpu.program);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gpgpu.indexBuffer);
      bindVertexProgramAttributeStreams(gl, gpgpu.program, gpgpu.vertexBuffer);
    }
  }
  whichState = 'tf'
}

export function clamp(x, min, max) {
  return Math.min(Math.max(x, min), max)
}

export function lerp(x, min, max) {
  return min + (max - min) * x
}

export async function threeMode() {
  if (playModesSafe || whichState !== 'three') {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    renderer.resetState()
  }
  whichState = 'three'
}


export async function imgUrlToTensor(url) {
  const img = document.createElement("img")
  img.width = 224
  img.height = 224
  img.src = url
  const result = await (new Promise((resolve) => {
    img.onload = () => {
      const tensor = tf.browser.fromPixels(img, 3)
      const shaped = imagenetPreprocess(tensor.expandDims(0))
      // const shaped = tf.reverse(tf.add(tf.mul(tensor.expandDims(0), 1 / 127.5), -1), -1)
      resolve(shaped)
    }
  }))
  return result
}

export function dispose(tensor) {
  backend.disposeData(tensor.dataId)
}

export async function toUint8Array(tensor) {
  return new Uint8Array(await tensor.mul(tf.scalar(255)).data())
}

export function tensorInternalTexture(tensor) {
  const texData = tf.backend().texData.get(tensor.dataId)
  const tex = texData.texture
  return tex
}

export function decodeTensor(tensor) {
  return backend.decode(tensor.dataId)
}

export function decodedInternalTexture(tensor) {
  return tensorInternalTexture(backend.decode(tensor.dataId))
}

export function iTexOfPlane(plane) {
  return threeInternalTexture(plane.material.map)
}

export function tensorTextureGl(tensor) {
  let actExpanded = tf.depthToSpace(tf.expandDims(tf.tile(tensor.reverse(1).reverse(2), [4, 1, 1]), 0), 2, 'NCHW')
  // decodeTensor(actExpanded)
  const tensInternal = tensorInternalTexture(actExpanded)
  return tensInternal
}

export function tensorThreeTextureGl(tensor) {
  const tex = tensorTextureGl(tensor)
  const threeTex = new THREE.Texture()
  const internals = renderer.properties.get(threeTex)
  internals.__webglTexture = tex
  return threeTex
}

export function showActivationPlane(activation, plane) {
  const tensInternal = tensorTextureGl(activation)
  const texInternal = iTexOfPlane(plane)
  // console.log(actExpanded.shape)
  commonCopyTexture(tensInternal, texInternal, activation.shape[0], activation.shape[1])
}

export function showActivationAcrossPlanes(activation, planes, channelsLast = false, rgb = false) {
  if (channelsLast) {
    tf.tidy(() => {
      activation = tf.mul(tf.squeeze(activation), 0.5)
      const shape = activation.shape
      if (rgb) {
        let activationPadded = activation
        if (activation.shape[2] < planes.length) {
          activationPadded = tf.concat([activation, tf.zeros([activation.shape[0], activation.shape[1], planes.length * 3 - activation.shape[2]])], 2)
        } else if (activation.shape[2] > planes.length) {
          activationPadded = tf.slice(activation, [0, 0, 0], [activation.shape[0], activation.shape[1], planes.length * 3])
        }
        const layers = tf.split(tf.transpose(activationPadded, [2, 0, 1]), planes.length, 0)
        //.map(t => tf.concat([t, tf.ones([shape[0], shape[1], 1])], 2))
        for (let i = 0; i < planes.length; i++) {
          const plane = planes[i]
          const texInternal = iTexOfPlane(plane)
          if (!texInternal) continue;
          const layer = layers[i]
          decodeTensor(layer)
          console.log("copying rgb")
          // const colors = tf.split(layer, 3, 0)
          console.log(layer.shape)
          let layerExpanded = tf.depthToSpace(tf.expandDims(tf.pad(layer, [[0, 1], [0, 0], [0, 0]], 255)), 2, 'NCHW')
          console.log(layerExpanded.shape)
          console.log("printing rgb")
          const tensInternal = tensorInternalTexture(layerExpanded)
          commonCopyTexture(tensInternal, texInternal, shape[0] * 2, shape[1] * 2, true)
        }
      } else {
        const layers = tf.split(tf.transpose(tf.slice(activation, [0, 0, 0], [activation.shape[0], activation.shape[1], planes.length]), [2, 0, 1]), planes.length, 0)
        for (let i = 0; i < planes.length; i++) {
          const plane = planes[i]
          const texInternal = iTexOfPlane(plane)
          if (!texInternal) continue;
          const layer = layers[i]
          let layerExpanded = tf.depthToSpace(tf.expandDims(tf.tile(layer, [4, 1, 1]), 0), 2, 'NCHW')
          // decodeTensor(layerExpanded)
          const tensInternal = tensorInternalTexture(layerExpanded)
          // console.log(layerExpanded.shape)
          commonCopyTexture(tensInternal, texInternal, shape[0], shape[1])
        }
      }
    })
  }
}

export function threeTfTextureShaderMaterial(tensor) {
  const glTex = gl.createTexture()
  return new THREE.ShaderMaterial({
    uniforms: { tfTexture: { value: glTex } },
    vertexShader: `
    attribute vec2 uv;
		varying vec2 vUv;
			void main() {
        vUv = uv;
				gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

			}`,
    fragmentShader: `
    uniform sampler2D tfTexture;
      varying vec2 vUv;
			void main() {

				gl_FragColor = texture2D(tfTexture, vUv);

			}`,
  })
}

export function tensorTexture(tensor, useInt = false) { // @SWITCHY
  if (tensor.shape.length !== 3) {
    throw new Error(`image tensor needs to have 3 dims, is shape ${JSON.stringify(tensor.shape)}`)
  }
  const textureFormat = [null, THREE.LuminanceFormat, null, THREE.RGBFormat, THREE.RGBAFormat][tensor.shape[tensor.shape.length - 1]]
  // console.log(tensor.shape[tensor.shape.length - 1], textureFormat)
  let texture;
  tfMode()
  let array = tf.transpose(imagenetUnPreprocess(tensor), [2, 0, 1]).dataSync()
  if (useInt) array = new Uint8Array(array)
  threeMode()
  if (useInt) {
    texture = new THREE.DataTexture(array, tensor.shape[0], tensor.shape[1], textureFormat);
  } else {
    texture = new THREE.DataTexture(array, tensor.shape[0], tensor.shape[1], textureFormat, THREE.FloatType);
  }
  texture.generateMipmaps = false;
  tfMode()
  return texture
}

export function arrayTexture(arr, width, height) {
  return new THREE.DataTexture(array, width, height, RGBAFormat);
}

// models are imported from tf.keras.applications
// preprocessing from https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
export function imagenetPreprocess(tensor) {
  return tf.reverse(tf.sub(tensor, 95), -1)
}

export function imagenetUnPreprocess(tensor) {
  return tf.reverse(tf.add(tensor, 95), -1)
}

function normalizeImage(tensor, means, stds) {
  const subtensor = tf.tensor([[[means]]])
  const divtensor = tf.tensor([[[stds]]])
  return tf.div(tf.sub(tensor, subtensor), divtensor)
}

export async function tensorImagePlane(tensor, opacity = 1) {
  const texture = tensorThreeTextureGl(tensor)
  const plane = doubleSidedPlane(texture, opacity)
  return plane
}

export function tensorToArray(tensor) {
  const decoded = tensorInternalTexture(tensor)
  // const decoded = decodedInternalTexture(tensor)
  console.log(tensor)
  console.log(decoded)
  const arr = new Float32Array(tensor.size)
  const size = tensor.shape.reduce((a, b) => a * b)
  const flatSize = Math.ceil(size / 4)
  const w = Math.ceil(Math.sqrt(flatSize))
  const h = Math.ceil(flatSize / w)
  withAsFramebuffer(gl, decoded, w, h, () => {
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, arr)
  })
  return arr
}

export function doubleSidedPlane(texture, opacity = 1) {
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    opacity,
    side: THREE.DoubleSide,
    // color: 0x00ff00,
    transparent: opacity !== 1,
    // blending: THREE.AdditiveBlending,
  });
  // make one visible from front and one from back
  const object2 = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material)
  object2.rotateY(Math.PI)
  return object2
}

export function imagePlane(url, callback) {
  const loader = new THREE.TextureLoader();
  // load a resource
  loader.load(
    // resource URL
    url,
    // onLoad callback
    function (texture) {
      // texture.generateMipmaps = false;
      callback(doubleSidedPlane(texture))
    },
    // onProgress callback currently not supported
    undefined,
    // onError callback
    function (err) {
      console.error('An error happened.');
    }
  );
}


function bindVertexBufferToProgramAttribute(
  gl, program, attribute,
  buffer, arrayEntriesPerItem, itemStrideInBytes,
  itemOffsetInBytes) {
  const loc = gl.getAttribLocation(program, attribute);
  if (loc === -1) {
    // The GPU compiler decided to strip out this attribute because it's unused,
    // thus no need to bind.
    return false;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.vertexAttribPointer(
    loc, arrayEntriesPerItem, gl.FLOAT, false, itemStrideInBytes,
    itemOffsetInBytes);
  gl.enableVertexAttribArray(loc);
  return true;
}

function bindVertexProgramAttributeStreams(
  gl, program, vertexBuffer) {
  const posOffset = 0;               // x is the first buffer element
  const uvOffset = 3 * 4;            // uv comes after [x y z]
  const stride = (3 * 4) + (2 * 4);  // xyz + uv, each entry is 4-byte float.
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer)
  const success = bindVertexBufferToProgramAttribute(
    gl, program, 'clipSpacePos', vertexBuffer, 3, stride, posOffset);

  return success &&
    bindVertexBufferToProgramAttribute(
      gl, program, 'uv', vertexBuffer, 2, stride, uvOffset);
}

export function actStats(activation) {
  const { mean, variance } = tf.moments(activation)
  return { mean: mean.dataSync()[0], variance: variance.dataSync()[0] }
}