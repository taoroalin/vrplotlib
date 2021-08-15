import * as THREE from 'three';
import { tensorImagePlane, imgUrlToTensor, tfMode, threeMode, imagePlane, commonCopyTexture, threeInternalTexture, showActivationAcrossPlanes } from "./common.mjs";
import * as common from "./common.mjs";
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet"
import { imagenetLabels } from "./labels"
export class NetVis {
  // when I add the ability to modify activations, I'll do it by 
  static async create(world, canvas, config) {
    const thiss = new NetVis()
    thiss.transparency = 0.2
    thiss.canvas = canvas
    thiss.channelsLast = false;
    const model = await tf.loadLayersModel(config.url)
    thiss.world = world
    thiss.group = new THREE.Group()
    world.add(thiss.group)
    thiss.group.rotation.y += 0.75
    thiss.group.position.x -= 1
    thiss.group.position.z -= 8
    thiss.topPredictions = []
    console.log(model)
    thiss.outputLayers = []
    for (let layer of model.layers) {
      if (layer.name.match(/conv\d?d?$/)) {
        if (!thiss.channelsLast && layer.dataFormat === "channelsLast") {
          thiss.channelsLast = true
        }
        thiss.outputLayers.push(layer.outboundNodes[0].outputTensors[0])
      }
    }
    const modelspec = { inputs: model.inputs, outputs: [...thiss.outputLayers, model.outputs[0]] }
    console.log(modelspec)
    thiss.model = tf.model(modelspec)
    console.log(thiss.model)
    thiss.activationPlaneGroups = []
    thiss.widthScale = 1 / 50
    thiss.inputShape = model.feedInputShapes[0]
    thiss.inputShape[0] = 1
    thiss.inputPlane = await tensorImagePlane(tf.zeros(thiss.inputShape))
    thiss.inputPlane.scale.x = thiss.inputShape[1] * thiss.widthScale
    thiss.inputPlane.scale.y = thiss.inputShape[1] * thiss.widthScale
    thiss.inputPlane.scale.z = thiss.inputShape[1] * thiss.widthScale
    thiss.group.add(thiss.inputPlane)
    thiss.maxPlanes = 20
    let visDepth = 0.1
    for (let output of thiss.outputLayers) {
      const actShape = output.shape
      const numPlanes = Math.min(Math.ceil(actShape[3] / 3), thiss.maxPlanes)
      const planes = []
      const activationGroup = new THREE.Group()
      thiss.group.add(activationGroup)
      const planeShape = thiss.channelsLast ? [actShape[1], actShape[2], 3] : [3, actShape[2], actShape[3]];
      for (let i = 0; i < numPlanes; i++) {
        const plane = await tensorImagePlane(tf.zeros(planeShape), thiss.transparency)
        plane.position.z += visDepth
        plane.scale.x = actShape[1] * thiss.widthScale
        plane.scale.z = actShape[1] * thiss.widthScale
        plane.scale.y = actShape[1] * thiss.widthScale
        activationGroup.add(plane)
        planes.push(plane)
        visDepth += 0.05
      }
      thiss.activationPlaneGroups.push(planes)
      visDepth += 0.25
    }

    thiss.pixelSelectObj = new THREE.Mesh(new THREE.BoxGeometry(thiss.widthScale, thiss.widthScale, thiss.widthScale * 10), new THREE.MeshBasicMaterial({ color: 0x00ff00 }))
    thiss.group.add(thiss.pixelSelectObj)
    thiss.pixelSelectObj.position.z -= thiss.widthScale

    thiss.updating = false
    thiss.inputTensor = tf.zeros(thiss.inputShape)
    thiss.activationTensors = []
    thiss.delay = 4
    thiss.lastUpdate = -9999999999

    const images = (await Promise.all(["n01440764_tench.jpeg", "n01443537_goldfish.jpeg", "n01484850_great_white_shark.jpeg", "n01491361_tiger_shark.jpeg", "n01494475_hammerhead.jpeg"].map(url => imgUrlToTensor("./imagenet/" + url)))).map(x => x.resizeBilinear([thiss.inputShape[1], thiss.inputShape[2]]))
    thiss.images = images
    thiss.imageIndex = 0
    console.log("HEREs the IMAGE", await images[0].data())
    thiss.inputTensor = images[0]
    thiss.selectedActivationIndex = 0
    thiss.selectedPlaneIndex = 0
    thiss.selectedPixel = [0, 0]
    thiss.deselectedTransparency = 0.025
    thiss.setSelected(0, 0)
    thiss.translateSelectedPixel(0, 0)

    thiss.doCycleImage = true
    thiss.setupListeners()

    return thiss
  }

  translateSelectedPixel(dx, dy) {
    this.selectedPixel[0] = Math.min(Math.max(this.selectedPixel[0] + dx, 0), this.inputShape[1])
    this.selectedPixel[1] = Math.min(Math.max(this.selectedPixel[1] + dy, 0), this.inputShape[2])

  }

  setupListeners() {
    document.addEventListener("keydown", (event) => {
      event.preventDefault()
      let caught = true
      if (event.shiftKey) {
        switch (event.key) {
          case "ArrowRight":
            this.group.rotation.y -= 0.05
            break
          case "ArrowLeft":
            this.group.rotation.y += 0.05
            break
        }
      } else if (event.ctrlKey) {
        switch (event.key) {
          case "ArrowRight":
            this.translateSelectedPixel(1, 0)
            break;
          case "ArrowLeft":
            this.translateSelectedPixel(-1, 0)
            break;
          case "ArrowUp":
            this.translateSelectedPixel(0, 1)
            break;
          case "ArrowDown":
            this.translateSelectedPixel(0, -1)
            break;
          default:
            caught = false;
        }
      } else {

        switch (event.key) {
          case "ArrowRight":
            this.setSelected(this.selectedActivationIndex, this.selectedPlaneIndex + 1)
            break;
          case "ArrowLeft":
            this.setSelected(this.selectedActivationIndex, this.selectedPlaneIndex - 1)
            break;
          case "ArrowUp":
            this.setSelected(this.selectedActivationIndex + 1, this.selectedPlaneIndex)
            break;
          case "ArrowDown":
            this.setSelected(this.selectedActivationIndex - 1, this.selectedPlaneIndex)
            break;
          default:
            caught = false;
        }
      }
      if (caught) {
        event.preventDefault()
      }
    })
  }

  setSelected(ai, pi) {
    const oldGroup = this.activationPlaneGroups[this.selectedActivationIndex]
    for (let i = 0; i < oldGroup.length; i++) {
      const oldPlane = oldGroup[i]
      oldPlane.children[0].material.opacity = this.transparency
    }
    const oldPos = oldGroup[0].position.z

    this.selectedActivationIndex = Math.min(Math.max(ai, 0), this.activationPlaneGroups.length - 1)
    const newGroup = this.activationPlaneGroups[this.selectedActivationIndex]
    this.selectedPlaneIndex = (pi + newGroup.length) % (newGroup.length)
    const newPos = newGroup[0].position.z
    this.group.translateZ(oldPos - newPos)
    for (let i = 0; i < newGroup.length; i++) {
      const plane = newGroup[i]
      plane.children[0].material.opacity = this.deselectedTransparency
    }
    const plane = newGroup[this.selectedPlaneIndex]
    plane.children[0].material.opacity = 1
  }

  cycleImage() {
    this.imageIndex = (this.imageIndex + 1) % this.images.length
    this.inputTensor = this.images[this.imageIndex]
  }

  async display() {
    tf.tidy(() => {
      showActivationAcrossPlanes(this.inputTensor, [this.inputPlane], this.channelsLast)
      for (let i = 0; i < this.activationTensors.length; i++) {
        const activation = this.activationTensors[i]
        const planes = this.activationPlaneGroups[i]
        showActivationAcrossPlanes(tf.mul(activation, 2), planes, this.channelsLast)
      }
    })
    // this.activationTensors[0].data().then(x => console.log("activation 1", x))
    // this.activationTensors[10].data().then(x => console.log("activation 10", x))
  }

  async _update() {
    if (this.doCycleImage) this.cycleImage()
    this.activationTensors.forEach(t => t.dispose())
    const pstime = performance.now()
    this.activationTensors = this.model.predict(this.inputTensor)
    this.probs = this.activationTensors.pop()
    this.probs.data().then((data) => {
      console.log(data)
      const zipped = Array.from(data).map((x, i) => {
        return [x, i]
      })
      zipped.sort((a, b) => b[0] - a[0])
      console.log(zipped)
      for (let i = 0; i < 1; i++) {
        console.log(imagenetLabels[zipped[i][1]])
      }
    })
    console.log(this.probs)
    console.log(`predict took ${performance.now() - pstime}`)
    await this.display()
  }

  update(inputs) {
    this.userInputs = inputs
    if (!this.updating && (this.lastUpdate + this.delay * 1000 < performance.now())) {
      this.updating = true
      this.lastUpdate = performance.now()
      this._update().then(() => {
        this.updating = false
      })
    }
  }
}