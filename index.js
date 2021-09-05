import * as twgl from "twgl.js";
import * as THREE from 'three';
import { OrbitControls } from './OrbitControls.js';
import { VRButton } from './VRAuto.js';
import { XRControllerModelFactory } from './example-webxr/XRControllerModelFactory.mjs';

import { imagePlane, setRendererAndTf, tfMode, threeMode } from "./common.mjs";

import { NetVis } from "./netvis.mjs";

import { setWebGLContext } from "@tensorflow/tfjs-backend-webgl";
import * as tf from "@tensorflow/tfjs";
let enableVRImmediately = false;

let visualization;

let container, canvas;
let camera, scene, renderer;
let controllerLeft, controllerRight;
let controllerGripLeft, controllerGripRight;

let grabposLeft, grabposRight;

let raycaster;

const intersected = [];
const tempMatrix = new THREE.Matrix4();

let controls, world, group;

let rotpoint;

let glContext;


const config = { moveScale: true }

let prevButtons = { left: {}, right: {} }
let buttons, axes;

const tempfn = async () => {

  await init();

  // imagePlane("./imagenet/n09468604_valley.JPEG", (object) => {
  //   world.add(object)
  //   object.position.add(new THREE.Vector3(1, 1, 1))
  // })
  visualization = await NetVis.create(world, canvas, { models: "./models", name: "ResNet50", input: "n01514668_cock.jpeg", images: "./imagenet", deepdream: "./deepdream" })
  window.vis = visualization
  // visualization = await NetVis.create(world, { url: "https://tfhub.dev/google/tfjs-model/imagenet/inception_v2/feature_vector/3/default/1" })
  animate();
}
tempfn()

window.addEventListener('vrdisplayactivate', function () {
  console.log("vrdisplayactivate")
});

async function init() {

  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  glContext = renderer.getContext()
  setWebGLContext(2, glContext);
  await tf.setBackend('webgl');
  setRendererAndTf(renderer)

  container = document.createElement('div');
  document.body.appendChild(container);

  scene = new THREE.Scene();
  world = new THREE.Group();
  world.position.add(new THREE.Vector3(0, 1, 0.5))
  scene.add(world);
  scene.background = new THREE.Color(0x808080);

  // this is flatscreen camera - xr makes camera with different settings
  camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.07, 300);
  camera.position.set(0, 0.6, 0);

  controls = new OrbitControls(camera, container);
  controls.target.set(0, 0.6, 0);
  controls.update();

  const useFloor = false
  if (useFloor) {
    const floorGeometry = new THREE.PlaneGeometry(16, 16);
    const floorMaterial = new THREE.MeshStandardMaterial({
      color: 0xeeeeee,
      roughness: 1.0,
      metalness: 0.0
    });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = - Math.PI / 2;
    floor.position.y -= 1
    floor.receiveShadow = true;
    world.add(floor);
  }

  world.add(new THREE.HemisphereLight(0x808080, 0x606060));

  const light = new THREE.DirectionalLight(0xffffff);
  light.position.set(0, 6, 0);
  light.castShadow = true;
  light.shadow.camera.top = 2;
  light.shadow.camera.bottom = - 2;
  light.shadow.camera.right = 2;
  light.shadow.camera.left = - 2;
  light.shadow.mapSize.set(4096, 4096);
  world.add(light);


  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.shadowMap.enabled = true;
  renderer.xr.enabled = true;
  container.appendChild(renderer.domElement);
  canvas = renderer.domElement

  document.body.appendChild(VRButton.createButton(renderer, enableVRImmediately));

  // controllers
  const contone = renderer.xr.getController(0);
  const conttwo = renderer.xr.getController(1);
  const gripone = renderer.xr.getControllerGrip(0);
  const griptwo = renderer.xr.getControllerGrip(1);
  // @TODO: make sure controller 0 is actually left
  controllerLeft = contone
  // controllerLeft.addEventListener('selectstart', onSelectStart);
  // controllerLeft.addEventListener('selectend', onSelectEnd);
  scene.add(controllerLeft);

  controllerRight = conttwo
  // controllerRight.addEventListener('selectstart', onSelectStart);
  // controllerRight.addEventListener('selectend', onSelectEnd);
  scene.add(controllerRight);

  const controllerModelFactory = new XRControllerModelFactory();

  controllerGripLeft = gripone
  controllerGripLeft.add(controllerModelFactory.createControllerModel(controllerGripLeft));
  scene.add(controllerGripLeft);

  controllerGripRight = griptwo
  controllerGripRight.add(controllerModelFactory.createControllerModel(controllerGripRight));
  scene.add(controllerGripRight);

  //

  const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, - 1)]);

  const line = new THREE.Line(geometry);
  line.name = 'line';
  line.scale.z = 5;

  controllerLeft.add(line.clone());
  controllerRight.add(line.clone());

  raycaster = new THREE.Raycaster();

  window.addEventListener('resize', onWindowResize);

}

function onWindowResize() {

  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);

}

function animate() {
  // render once to make three load the textures
  renderer.render(scene, camera);
  renderer.setAnimationLoop(render);
}

let isAnimating = true
document.addEventListener("keydown", (event) => {
  switch (event.key) {
    case "c":
      if (event.ctrlKey) {
        if (isAnimating) {
          isAnimating = false
          renderer.setAnimationLoop(null)
        } else {
          isAnimating = true
          renderer.setAnimationLoop(render)
        }
        event.preventDefault()
      }
      break
  }
})

function readInputs() {
  buttons = { left: [], right: [] }
  axes = { left: [], right: [] }
  const session = renderer.xr.getSession();
  if (session && session.inputSources) {  //only if we are in a webXR session
    for (const sourceXR of session.inputSources) {
      console.log(sourceXR)
      if (sourceXR.gamepad) {
        buttons[sourceXR.handedness] = sourceXR.gamepad.buttons
        axes[sourceXR.handedness] = sourceXR.gamepad.axes
      } else {
        console.error("sourceXR.gamepad not found (can't use buttons)")
      }
    }
  }
}

function grabMovement() {
  // gorn+tiltbrush movement
  if (buttons.right[4]?.pressed && buttons.left[4]?.pressed) {
    const prevDir = grabposLeft.clone().sub(grabposRight)
    const newDir = controllerLeft.position.clone().sub(controllerRight.position)

    const angle = new THREE.Quaternion();
    angle.setFromUnitVectors(prevDir.clone().setY(0).normalize(), newDir.clone().setY(0).normalize())

    if (config.moveScale) {

      const scalar = newDir.length() / prevDir.length()
      const mat = new THREE.Matrix4()
      mat.makeScale(scalar, scalar, scalar)
      const offset = controllerRight.position.clone().sub(world.position)

      // scale AROUND SCENE ORIGIN and correct to around controller
      world.applyMatrix4(mat)
      world.position.sub(controllerRight.position.clone().multiplyScalar(scalar - 1))

      // translate, rotate, translate
      world.position.add(offset)
      world.applyQuaternion(angle)
      world.position.sub(offset.applyQuaternion(angle))

      // move
      world.position.add(controllerRight.position).sub(grabposRight)
    } else {

      const prevMidpoint = prevDir.clone().multiplyScalar(0.5).add(grabposRight)
      const midpoint = newDir.clone().multiplyScalar(0.5).add(controllerRight.position)

      // rotate world around controller midpoint
      const oldPosition = world.position.clone()
      const offset = midpoint.clone().sub(oldPosition)
      world.position.add(offset)
      world.applyQuaternion(angle)
      world.position.sub(offset.applyQuaternion(angle))

      world.position.add(midpoint).sub(prevMidpoint)
    }
  } else if (buttons.right[4]?.pressed) {
    const delta = controllerRight.position.clone().sub(grabposRight)
    world.position.add(delta)
  } else if (buttons.left[4]?.pressed) {
    const delta = controllerLeft.position.clone().sub(grabposLeft)
    world.position.add(delta)
  }
  grabposLeft = controllerLeft.position.clone()
  grabposRight = controllerRight.position.clone()
}

function render() {
  twgl.resizeCanvasToDisplaySize(canvas)
  threeMode()

  readInputs()

  grabMovement()

  const inputs = { controllerLeft, controllerRight, buttons, prevButtons, axes }
  if (visualization) {
    visualization.update(inputs)
  }
  renderer.render(scene, camera);
  prevButtons = JSON.parse(JSON.stringify(buttons))
}
