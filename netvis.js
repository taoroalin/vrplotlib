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
    this.verticalSpacing = 0.2
    this.horizontalSpacing = 0.2
    this.canvas = canvas
    this.channelsLast = false;
    this.uniformFilterSize = true

    this.spec = { layers: {}, focusedLayer: null, zoom: 1, injected: {}, input: config.input, name: config.name, cameraLocked: true, }

    this.dirs = { models: config.models, images: config.images, deepdream: config.deepdream }
    const url = this.dirs.models + "/" + this.spec.name + "/model.json"
    this.dreamurl = this.dirs.deepdream + `/filter/${this.spec.name}/`
    this.dreamcache = {}
    console.log(url)

    const model = await tf.loadLayersModel(url)
    console.log(model)
    this.world = world
    this.group = new THREE.Group()
    this.group.name = ("netvis")
    world.add(this.group)
    this.group.position.z -= 8
    this.group.position.x = 0
    this.group.position.y = 0

    this.activationsGroup = new THREE.Group()
    this.activationsGroup.name = "activations"
    this.group.add(this.activationsGroup)

    this.topPredictions = []
    this.outputLayers = []
    for (let layer of model.layers) {
      if (layer.name.match(/conv\d?d?$/) && !layer.name.match(/bn$/)) {
        if (!this.channelsLast && layer.dataFormat === "channelsLast") {
          this.channelsLast = true
        }
        const symTensor = layer.outboundNodes[0].outputTensors[0]
        this.outputLayers.push(symTensor)
      }
    }
    const modelspec = { inputs: model.inputs, outputs: [...this.outputLayers, model.outputs[0]] }
    this.model = tf.model(modelspec)
    this.inputShape = model.feedInputShapes[0]
    this.inputShape[0] = 1

    for (let output of this.model.outputs) {
      this.spec.layers[this.actName(output)] = { show: true, fv: false, shownFilters: [1], activeFilter: 1, shape: output.shape, _: {} }
    }

    this.widthScale = 1 / 50
    this.sideSpacing = 1.5

    this.fontSize = 0.15
    this.labelOffset = 0.3

    let side = this.inputShape[1] * this.widthScale * this.sideSpacing
    for (let li = 0; li < this.outputLayers.length; li++) {
      const output = this.outputLayers[li]
      const actShape = output.shape
      const planes = []

      const activationGroup = new THREE.Group()
      activationGroup.name = this.actName(output)
      this.activationsGroup.add(activationGroup)
      activationGroup.position.x += side
      side += actShape[1] * this.widthScale * this.sideSpacing

      const activationLabel = new Text()
      activationGroup.add(activationLabel)
      activationLabel.name = "label"
      const filtersGroup = new THREE.Group()
      filtersGroup.name = "filters"
      activationGroup.add(filtersGroup)

      const outputName = this.actName(output) // .replaceAll(/(^.+\/)|(_bn)/g, "")
      const approxLength = outputName.length * this.fontSize
      activationLabel.text = outputName
      activationLabel.fontSize = this.fontSize
      activationLabel.color = 0xFFFFFF

      activationLabel.position.y -= actShape[1] * this.widthScale * 0.5 + this.labelOffset
      activationLabel.position.x -= approxLength * 0.27

      activationLabel.sync()
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

    this.inputPlane = await tensorImagePlane(this.inputTensor.slice([0, 0, 0, 0], [1, -1, -1, 1]).squeeze(0), true)
    this.inputPlane.scale.x = this.inputShape[1] * this.widthScale * 0.5
    this.inputPlane.scale.y = this.inputShape[1] * this.widthScale * 0.5
    this.inputPlane.scale.z = this.inputShape[1] * this.widthScale * 0.5
    this.inputPlane.position.z = 8
    this.inputPlane.position.y = -6
    this.world.add(this.inputPlane)


    this.setupListeners()

    console.log(this.spec)

    return this
  }

  actName(x) {
    return x.name.match(/^[^/]+/)[0].replace("_bn", "_conv")
    // return x.name.match(/^[^/]+/)[0]
  }

  async getDream(layerName, idx) {
    const cachename = layerName + "." + idx
    const url = this.dreamurl + layerName + "/" + idx + ".png"
    if (this.dreamcache[cachename] !== undefined) {
      return this.dreamcache[cachename]
    } else {
      const plane = await new Promise(resolve => imagePlane(url, resolve))
      this.dreamcache[cachename] = plane
      return plane
    }
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
    const layersGroup = this.group.getObjectByName('activations')
    let xposition = 0;
    let skipped = false
    for (let layerName in this.spec.layers) {
      if (layerName === "predictions") continue
      if (!skipped) {
        skipped = true;
        // continue
      }
      const layer = this.spec.layers[layerName]
      if (layer.show) {
        const layerGroup = layersGroup.getObjectByName(layerName)
        const filtersGroup = layerGroup.getObjectByName("filters")
        const output = this.activationTensors[layerName]

        layerGroup.visible = true;
        const layerShape = common.debatchShape(layer.shape)
        const scale = this.uniformFilterSize ? 1 : layer.shape[1] * this.widthScale;
        const xtaken = scale + scale + this.horizontalSpacing * 2
        layerGroup.position.x = xposition
        xposition += xtaken
        for (let i = filtersGroup.children.length; i < layer.shownFilters.length; i++) {
          const filterGroup = new THREE.Group()
          filtersGroup.add(filterGroup)
          const plane = await tensorImagePlane(tf.zeros(layerShape))
          filterGroup.add(plane)
          plane.name = "filter"
          plane.position.z += i * 0.05
          if (!this.uniformFilterSize) {
            plane.scale.x = layerShape[1] * this.widthScale
            plane.scale.z = layerShape[1] * this.widthScale
            plane.scale.y = layerShape[1] * this.widthScale
          }
        }
        for (let i = layer.shownFilters.length; i < filtersGroup.children.length; i++) {
          const plane = filtersGroup.children[i]
          plane.visible = false
        }
        const offset = layer.shownFilters.findIndex(x => x === layer.activeFilter)
        const height = scale + this.verticalSpacing
        let zposition = -height * offset
        for (let i = 0; i < layer.shownFilters.length; i++) {
          const filter = layer.shownFilters[i]
          const filterGroup = filtersGroup.children[i]
          const filterPlane = filterGroup.getObjectByName("filter")
          const filterTensor = tf.slice(output, [0, 0, 0, filter], [-1, -1, -1, 1]).squeeze(0)
          filterGroup.position.z = zposition
          zposition += height
          // console.log(filterTensor.dataSync())
          common.showActivationPlane(filterTensor, filterPlane)
          this.getDream(layerName, filter).then(dream => {
            if (dream) {
              filterGroup.add(dream)
              dream.position.x = scale + this.horizontalSpacing
            }
          })
          if (this.spec.cameraLocked && layer.focusedFilter == filter) {
            this.world.position.z = zposition
          }
        }
        if (this.spec.cameraLocked && this.spec.focusedLayer == layerName) {
          this.world.position.x = xposition
        }
      } else {
        layerGroup.visible = false;
      }
    }
  }

  async _update() {
    for (let k in this.activationTensors) {
      const t = this.activationTensors[k]
      if (t) t.dispose()
    }
    this.activationTensors = {}
    const pstime = performance.now()
    const at = this.model.predict(this.inputTensor)
    for (let i = 0; i < this.model.outputs.length; i++) {
      const output = this.model.outputs[i]
      this.activationTensors[this.actName(output)] = at[i]
    }
    console.log(this.activationTensors)
    this.probs = at[this.model.outputs.length - 1]
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
  selectNextLayerOrder(t) {
    const keys = Object.keys(this.spec.layers)
    const oldIndex = keys.indexOf(this.spec.focusedLayer)
    const newIndex = Math.min(Math.max(oldIndex + t, 0), keys.length - 1)
    const newKey = keys[newIndex]
    this.spec.focusedLayer = newKey
  }
  selectNextFilterOrder(t) {
    const layer = this.spec.layers[this.spec.focusedLayer]
    const newFilter = Math.min(Math.max(layer.shownFilters.indexOf(layer.focusedFilter) + t, 0), layer.shape[layer.shape.length - 1] - 1)
    layer.focusedFilter = newFilter
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
          case "ArrowUp":
            this.selectNextFilterOrder(1)
            break;
          case "ArrowDown":
            this.selectNextFilterOrder(-1)
            break;
          case "ArrowRight":
            this.selectNextLayerOrder(-1)
            break;
          case "Arrow:Left":
            this.selectNextLayerOrder(-1)
            break;
          default:
            caught = false;
        }
      }
      if (caught) {
        this.display()
        event.preventDefault()
      }
    })
  }
}