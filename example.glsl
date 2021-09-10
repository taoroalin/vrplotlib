  
  int getFlatIndex(ivec3 coords) {
    return coords.x * 896 + coords.y * 4 + coords.z;
  }


      void main() {
        ivec3 coords = getOutputCoords();

        vec4 result = vec4(0.);
        int flatIndex, r, c, offset;
        ivec3 localCoords;
        vec2 uv;
        vec4 values;

        
          localCoords = coords;
          if(localCoords[2] + 0 < 4) {
            localCoords[2] += 0;
            if(localCoords[1] + 0 < 224) {
              localCoords[1] += 0;

              flatIndex = getFlatIndex(localCoords);
              offset = imod(flatIndex, 4);

              flatIndex = idiv(flatIndex, 4, 1.);

              r = flatIndex / 448;
              c = imod(flatIndex, 448);
              uv = (vec2(c, r) + halfCR) / vec2(448.0, 112.0);
              values = texture(A, uv);

              if(offset == 0) {
                result[0] = values[0];
              } else if(offset == 1) {
                result[0] = values[1];
              } else if(offset == 2) {
                result[0] = values[2];
              } else {
                result[0] = values[3];
              }
            }
          }
        
          localCoords = coords;
          if(localCoords[2] + 1 < 4) {
            localCoords[2] += 1;
            if(localCoords[1] + 0 < 224) {
              localCoords[1] += 0;

              flatIndex = getFlatIndex(localCoords);
              offset = imod(flatIndex, 4);

              flatIndex = idiv(flatIndex, 4, 1.);

              r = flatIndex / 448;
              c = imod(flatIndex, 448);
              uv = (vec2(c, r) + halfCR) / vec2(448.0, 112.0);
              values = texture(A, uv);

              if(offset == 0) {
                result[1] = values[0];
              } else if(offset == 1) {
                result[1] = values[1];
              } else if(offset == 2) {
                result[1] = values[2];
              } else {
                result[1] = values[3];
              }
            }
          }
        
          localCoords = coords;
          if(localCoords[2] + 0 < 4) {
            localCoords[2] += 0;
            if(localCoords[1] + 1 < 224) {
              localCoords[1] += 1;

              flatIndex = getFlatIndex(localCoords);
              offset = imod(flatIndex, 4);

              flatIndex = idiv(flatIndex, 4, 1.);

              r = flatIndex / 448;
              c = imod(flatIndex, 448);
              uv = (vec2(c, r) + halfCR) / vec2(448.0, 112.0);
              values = texture(A, uv);

              if(offset == 0) {
                result[2] = values[0];
              } else if(offset == 1) {
                result[2] = values[1];
              } else if(offset == 2) {
                result[2] = values[2];
              } else {
                result[2] = values[3];
              }
            }
          }
        
          localCoords = coords;
          if(localCoords[2] + 1 < 4) {
            localCoords[2] += 1;
            if(localCoords[1] + 1 < 224) {
              localCoords[1] += 1;

              flatIndex = getFlatIndex(localCoords);
              offset = imod(flatIndex, 4);

              flatIndex = idiv(flatIndex, 4, 1.);

              r = flatIndex / 448;
              c = imod(flatIndex, 448);
              uv = (vec2(c, r) + halfCR) / vec2(448.0, 112.0);
              values = texture(A, uv);

              if(offset == 0) {
                result[3] = values[0];
              } else if(offset == 1) {
                result[3] = values[1];
              } else if(offset == 2) {
                result[3] = values[2];
              } else {
                result[3] = values[3];
              }
            }
          }
        

        outputColor = result;
      }
