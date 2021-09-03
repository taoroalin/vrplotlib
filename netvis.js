import * as THREE from 'three';
import { tensorImagePlane, imgUrlToTensor, imagePlane, showActivationAcrossPlanes } from "./common.mjs";
import * as common from "./common.mjs";
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet"
import { imagenetLabels } from "./labels"
import { Text } from 'troika-three-text'
export class NetVis {
  // when I add the ability to modify activations, I'll do it by 
  static async create(world, canvas, config) {

    const testtens = tf.tensor([[1, 2, 3], [5, 6, 7]])
    console.log(testtens)
    // const arr = common.tensorToArray(testtens)
    // console.log("ARRAY", arr)

    const thiss = new NetVis()
    await thiss.init(world, canvas, config)
    return thiss
  }

  async init(world, canvas, config) {
    this.transparency = 0.2
    this.canvas = canvas
    this.channelsLast = false;

    this.spec = { layers: {}, focusedLayer: null, injected: {}, input: config.input, name: config.name }

    this.dirs = { models: config.models, images: config.images, deepdream: config.deepdream }
    const url = this.dirs.models + "/" + this.spec.name + "/model.json"
    console.log(url)

    const model = await tf.loadLayersModel(url)
    this.world = world
    this.group = new THREE.Group()
    world.add(this.group)
    this.group.position.z -= 8
    this.group.position.x -= 7.5

    this.activationsGroup = new THREE.Group()
    this.group.add(this.activationsGroup)

    this.topPredictions = []
    this.outputLayers = []
    for (let layer of model.layers) {
      if (layer.name.match(/conv\d?d?$/)) {
        if (!this.channelsLast && layer.dataFormat === "channelsLast") {
          this.channelsLast = true
        }
        this.outputLayers.push(layer.outboundNodes[0].outputTensors[0])
      }
    }
    const modelspec = { inputs: model.inputs, outputs: [...this.outputLayers, model.outputs[0]] }
    this.model = tf.model(modelspec)
    this.inputShape = model.feedInputShapes[0]
    this.inputShape[0] = 1

    for (let output of this.model.outputs) {
      this.spec.layers[output.name] = { show: true, fv: false, shownFilters: [0], activeFilter: 0 }
    }

    this.widthScale = 1 / 50
    this.sideSpacing = 1.5
    this.inputPlane = await tensorImagePlane(tf.squeeze(tf.zeros(this.inputShape)), true)
    this.inputPlane.scale.x = this.inputShape[1] * this.widthScale
    this.inputPlane.scale.y = this.inputShape[1] * this.widthScale
    this.inputPlane.scale.z = this.inputShape[1] * this.widthScale
    this.group.add(this.inputPlane)

    this.fontSize = 0.15
    this.labelOffset = 0.3

    let side = this.inputShape[1] * this.widthScale * this.sideSpacing
    for (let li = 0; li < this.outputLayers.length; li++) {
      const output = this.outputLayers[li]
      const actShape = output.shape
      const numPlanes = Math.min(Math.ceil(actShape[3] / 3), this.maxPlanes)
      const planes = []

      const activationGroup = new THREE.Group()
      this.group.add(activationGroup)
      activationGroup.position.x += side
      side += actShape[1] * this.widthScale * this.sideSpacing

      const activationLabel = new Text()
      activationGroup.add(activationLabel)

      const outputName = output.name.replaceAll(/(^.+\/)|(_bn)/g, "")
      const approxLength = outputName.length * this.fontSize
      activationLabel.text = outputName
      activationLabel.fontSize = this.fontSize
      activationLabel.color = 0xFFFFFF

      activationLabel.position.y -= actShape[1] * this.widthScale * 0.5 + this.labelOffset
      activationLabel.position.x -= approxLength * 0.27

      activationLabel.sync()

      const planeShape = this.channelsLast ? [actShape[1], actShape[2], 1] : [1, actShape[2], actShape[3]];
      for (let i = 0; i < numPlanes; i++) {
        const plane = await tensorImagePlane(tf.zeros(planeShape), this.transparency)
        plane.position.z += i * 0.05
        plane.scale.x = actShape[1] * this.widthScale
        plane.scale.z = actShape[1] * this.widthScale
        plane.scale.y = actShape[1] * this.widthScale
        activationGroup.add(plane)
        planes.push(plane)
      }
      // this.activationPlaneGroups.push(planes)
    }

    this.pixelSelectObj = new THREE.Mesh(new THREE.BoxGeometry(this.widthScale, this.widthScale, this.widthScale * 10), new THREE.MeshBasicMaterial({ color: 0x00ff00 }))
    this.group.add(this.pixelSelectObj)
    this.pixelSelectObj.position.z -= this.widthScale

    this.updating = false
    this.activationTensors = {}
    this.delay = 4
    this.lastUpdate = -9999999999

    this.inputTensor = (await this.getImageTensor(this.spec.input)).resizeBilinear([this.inputShape[1], this.inputShape[2]])
    this.selectedActivationIndex = 0
    this.selectedPlaneIndex = 0
    this.selectedPixel = [0, 0]


    this.setupListeners()

    return this
  }

  createFilterVisual(symbolicTensor, idx) {
    const shape = symbolicTensor.shape
    const plane = tensorImagePlane()
  }

  async getImageTensor(name) {
    const url = this.dirs.images + "/" + name
    const t = await imgUrlToTensor(url)
    return t.resizeBilinear([this.inputShape[1], this.inputShape[2]])
  }

  //@STUCK I don't know of a way to alter the middle of a compute graph in tfjs
  async createActivationInjectedVis(activationIndex) {
    // FOR NOW THIS CORRUPTS OLD MODEL
    const thiss = new NetVis()
    const oldOutputs = this.model.outputs
    const newInputTensor = oldOutputs[activationIndex].outboundNodes[0].outputTensors[0]
    thiss.model = tf.model({ inputs: [...this.model.inputs, newInputTensor], outputs: oldOutputs.slice(activationIndex) })
    return thiss
  }

  getFeatureVisualizationPlane(name, number) {
    const url = `./deepdream/filter/${this.model.name}/${name}/${number}.jpg`
    const plane = imagePlane(url)
    return plane
  }

  translateSelectedPixel(dx, dy) {
    this.selectedPixel[0] = Math.min(Math.max(this.selectedPixel[0] + dx, 0), this.inputShape[1])
    this.selectedPixel[1] = Math.min(Math.max(this.selectedPixel[1] + dy, 0), this.inputShape[2])

  }

  async display_old() {
    tf.tidy(() => {
      showActivationAcrossPlanes(this.inputTensor, [this.inputPlane], this.channelsLast, true)
      for (let i = 0; i < this.activationTensors.length; i++) {
        const activation = this.activationTensors[i]
        const planes = this.activationPlaneGroups[i]
        showActivationAcrossPlanes(tf.mul(activation, 2), planes, this.channelsLast)
      }
    })
    // layer 0 supposed tpo be mean 0.6 variance 10,000
    // this.activationTensors[0].data().then(x => console.log("activation 1", x))
    // this.activationTensors[10].data().then(x => console.log("activation 10", x))
  }

  async display() {

    for (let layer in this.spec.layers) {
      if (layer.show) {
        console.log(layer)
        for (let filter in layer.filters) {

        }
      }
    }
  }

  async _update() {
    for (let k in this.activationTensors) {
      const t = this.activationTensors[k]
      t.dispose()
    }
    this.activationTensors = {}
    const pstime = performance.now()
    const at = this.model.predict(this.inputTensor)
    for (let i = 0; i < this.model.outputs.length; i++) {
      const output = this.model.outputs[i]
      this.activationTensors[output.name] = at[i]
      if (i === this.model.outputs.length - 1) this.probs = at[i]
    }
    const dstime = performance.now()
    const probsArray = this.probs.dataSync()
    // const arr = common.tensorToArray(this.probs)
    console.log('datasync took', performance.now() - dstime)
    const zipped = []
    for (let i = 0; i < probsArray.length; i++) {
      zipped.push([probsArray[i], i])
    }
    zipped.sort((a, b) => b[0] - a[0])

    for (let i = 0; i < 1; i++) {
      console.log(imagenetLabels[zipped[i][1]])
    }
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

  setupListeners() {
    document.addEventListener("keydown", (event) => {
      let caught = true
      if (event.shiftKey) {
        switch (event.key) {
          case "ArrowRight":
            this.group.rotation.y -= 0.05
            break
          case "ArrowLeft":
            this.group.rotation.y += 0.05
            break
          default:
            caught = false;
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
            this.setSelected(this.selectedActivationIndex - 1, this.selectedPlaneIndex)
            break;
          case "ArrowDown":
            this.setSelected(this.selectedActivationIndex + 1, this.selectedPlaneIndex)
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
}