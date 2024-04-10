#include "openGLHeader.h"
#include "glutHeader.h"
#include "openGLMatrix.h"
#include "imageIO.h"
#include "pipelineProgram.h"
#include "vbo.h"
#include "vao.h"
#include <glm/glm.hpp>


#include <iostream>
#include <cstring>
#include <vector>
#include <chrono>
#include <thread>

#if defined(WIN32) || defined(_WIN32)
  #ifdef _DEBUG
    #pragma comment(lib, "glew32d.lib")
  #else
    #pragma comment(lib, "glew32.lib")
  #endif
#endif

#if defined(WIN32) || defined(_WIN32)
  char shaderBasePath[1024] = SHADER_BASE_PATH;
#else
  char shaderBasePath[1024] = "../openGLHelper";
#endif

using namespace std;

int mousePos[2]; // x,y screen coordinates of the current mouse position

int leftMouseButton = 0; // 1 if pressed, 0 if not 
int middleMouseButton = 0; // 1 if pressed, 0 if not
int rightMouseButton = 0; // 1 if pressed, 0 if not

typedef enum { ROTATE, TRANSLATE, SCALE } CONTROL_STATE;
CONTROL_STATE controlState = ROTATE;

// Transformations of the terrain.
float terrainRotate[3] = { 0.0f, 0.0f, 0.0f };
float terrainTranslate[3] = { 0.0f, 0.0f, 0.0f };
float terrainScale[3] = { 1.0f, 1.0f, 1.0f };

// Width and height of the OpenGL window, in pixels.
int windowWidth = 1280;
int windowHeight = 720;
char windowTitle[512] = "CSCI 420 Homework 2";


OpenGLMatrix matrix;
PipelineProgram * pipelineProgram = nullptr;
PipelineProgram * texturePP = nullptr;

// more global variables
int imgH, imgW; // img height and width
float height;
int numVertices;
GLuint groundHandle;

float s = 0.5;
// 4x4 col-major basis matrix
double basis[16] = {-s,2*s,-s,0,2-s,s-3,0,1,s-2,3-2*s,s,0,s,-s,0,0};
bool ride = false, record = false, add = true;
int ind = 0, frame15 = 0, sc = 0;
float change = 1.0;
float eyeD[3] = {-12, 15, -12};
float centerD[3] = {23.5, 2.0, -12};
vector<float> splineP,splineC,tanP,normP,binoP,track1,track1c;
vector<float> ground,grounduv;
VBO* groundVBO;
VBO* grounduvVBO;
VAO* groundVAO;
VBO* track1VBO;
VBO* track1cVBO;
VAO* trackVAO;
VBO* normalsVBO;
VAO* normalsVAO;


void setTextureUnit(GLint unit)
{
 glActiveTexture(unit); // select texture unit affected by subsequent texture calls
 // get a handle to the “textureImage” shader variable
 GLint h_textureImage = glGetUniformLocation(texturePP->GetProgramHandle(),"textureImage");
 // deem the shader variable “textureImage” to read from texture unit “unit”
 glUniform1i(h_textureImage, unit - GL_TEXTURE0);
}
// helper function to get cross product
void cross(double x1,double y1,double z1,double x2,double y2,double z2,double local[3]){
  local[0] = y1 * z2 - z1 * y2;
  local[1] = z1 * x2 - x1 * z2;
  local[2] = x1 * y2 - y1 * x2;
}
// helper function to get unit vector
void unit(double P[3]){
  double mag = sqrt((P[0] * P[0]) + (P[1] * P[1]) + (P[2] * P[2]));
  P[0] = P[0] / mag;
  P[1] = P[1] / mag;
  P[2] = P[2] / mag;
}
// calculate the normals and binormals to each point
void calcUpVec(double tanx,double tany,double tanz,bool first){
  double V[3] = {0.0,1.0,0.0}; // V=(0,1,0)
  double local[3] = {0.0,0.0,0.0};
  if (first){ // N0 = unit(T0 x V), B0 = unit(T0 x N0)
    cross(tanx,tany,tanz,V[0],V[1],V[2],local);
    unit(local);
    normP.push_back(local[0]);
    normP.push_back(local[1]);
    normP.push_back(local[2]);
    cross(tanx,tany,tanz,local[0],local[1],local[2],V);
    unit(V);
    binoP.push_back(V[0]);
    binoP.push_back(V[1]);
    binoP.push_back(V[2]);
  } else { // N1 = unit(B0 x T1) and B1 = unit(T1 x N1)
    int curr = normP.size();
    cross(binoP[curr-3],binoP[curr-2],binoP[curr-1],tanx,tany,tanz,local);
    unit(local);
    normP.push_back(local[0]);
    normP.push_back(local[1]);
    normP.push_back(local[2]);
    cross(tanx,tany,tanz,local[0],local[1],local[2],V);
    unit(V);
    binoP.push_back(V[0]);
    binoP.push_back(V[1]);
    binoP.push_back(V[2]);
  }
}

void makeGround(){
  float bx = 100.0f; // boundary for ground
  float by = -100.0f;
  float bz = 100.0f;
  float groundPlane[] = {
    bx,by,bz,
    -bx,by,bz,
    -bx,by,-bz,
    bx,by,bz,
    bx,by,-bz,
    -bx,by,-bz
  };
  int length = sizeof(groundPlane) / sizeof(groundPlane[0]);
  for (int i=0;i<length;i++){
    ground.push_back(groundPlane[i]);
  }
  float groundplaneuv[] = {
    0,0,
    1,0,
    1,1,
    0,0,
    0,1,
    1,1
  };
  length = sizeof(groundplaneuv) / sizeof(groundplaneuv[0]);
  for (int i=0;i<length;i++){
    grounduv.push_back(groundplaneuv[i]);
  }
}

void makeTracks(){
  int curr = splineP.size();
  float alpha = 0.05; // find best fit value for viewing
  glm::vec3 p0(splineP[curr-6],splineP[curr-5],splineP[curr-4]);
  glm::vec3 p1(splineP[curr-3],splineP[curr-2],splineP[curr-1]);
  glm::vec3 n0(normP[curr-6],normP[curr-5],normP[curr-4]);
  glm::vec3 n1(normP[curr-3],normP[curr-2],normP[curr-1]);
  glm::vec3 b0(binoP[curr-6],binoP[curr-5],binoP[curr-4]);
  glm::vec3 b1(binoP[curr-3],binoP[curr-2],binoP[curr-1]);
  glm::vec3 V[8];
  V[0] = p0 + alpha * (-n0 + b0);
  V[1] = p0 + alpha * (n0 + b0);
  V[2] = p0 + alpha * (n0 - b0);
  V[3] = p0 + alpha * (-n0 - b0);
  V[4] = p1 + alpha * (-n1 + b1);
  V[5] = p1 + alpha * (n1 + b1);
  V[6] = p1 + alpha * (n1 - b1);
  V[7] = p1 + alpha * (-n1 - b1);
  int triangles[] = { // size 24, indices of 8 triangles for 4 faces
    0,1,5, // face 1, right
    0,4,5,
    1,2,6, // face 2, top
    1,5,6,
    2,3,7, // face 3, left
    2,6,7,
    3,0,4, // face 4, bottom
    3,7,4
  };
  glm::vec3 coloring[] = {
    b0,b0,b1, // coloring face 1, right
    b0,b1,b1,
    n0,n0,n1, // coloring face 2, top
    n0,n1,n1,
    -b0,-b0,-b1, // coloring face 3, left
    -b0,-b1,-b1,
    -n0,-n0,-n1, // coloring face 4, bottom
    -n0,-n1,-n1
  };
  for (int i=0;i<24;i++){
    track1.push_back(V[triangles[i]].x);
    track1.push_back(V[triangles[i]].y);
    track1.push_back(V[triangles[i]].z);
    track1c.push_back(coloring[i].x);
    track1c.push_back(coloring[i].y);
    track1c.push_back(coloring[i].z);
  }
}

// starter begin
// Represents one spline control point.
struct Point 
{
  double x, y, z;
};

// Contains the control points of the spline.
struct Spline 
{
  int numControlPoints;
  Point * points;
} spline;

void loadSpline(char * argv) 
{
  FILE * fileSpline = fopen(argv, "r");
  if (fileSpline == NULL) 
  {
    printf ("Cannot open file %s.\n", argv);
    exit(1);
  }

  // Read the number of spline control points.
  fscanf(fileSpline, "%d\n", &spline.numControlPoints);
  printf("Detected %d control points.\n", spline.numControlPoints);

  // Allocate memory.
  spline.points = (Point *) malloc(spline.numControlPoints * sizeof(Point));
  // Load the control points.
  for(int i=0; i<spline.numControlPoints; i++)
  {
    if (fscanf(fileSpline, "%lf %lf %lf", 
           &spline.points[i].x, 
	   &spline.points[i].y, 
	   &spline.points[i].z) != 3)
    {
      printf("Error: incorrect number of control points in file %s.\n", argv);
      exit(1);
    }
  }
}

// Multiply C = A * B, where A is a m x p matrix, and B is a p x n matrix.
// All matrices A, B, C must be pre-allocated (say, using malloc or similar).
// The memory storage for C must *not* overlap in memory with either A or B. 
// That is, you **cannot** do C = A * C, or C = C * B. However, A and B can overlap, and so C = A * A is fine, as long as the memory buffer for A is not overlaping in memory with that of C.
// Very important: All matrices are stored in **column-major** format.
// Example. Suppose 
//      [ 1 8 2 ]
//  A = [ 3 5 7 ]
//      [ 0 2 4 ]
//  Then, the storage in memory is
//   1, 3, 0, 8, 5, 2, 2, 7, 4. 
void MultiplyMatrices(int m, int p, int n, const double * A, const double * B, double * C)
{
  for(int i=0; i<m; i++)
  {
    for(int j=0; j<n; j++)
    {
      double entry = 0.0;
      for(int k=0; k<p; k++)
        entry += A[k * m + i] * B[j * p + k];
      C[m * j + i] = entry;
    }
  }
}

int initTexture(const char * imageFilename, GLuint textureHandle)
{
  // Read the texture image.
  ImageIO img;
  ImageIO::fileFormatType imgFormat;
  ImageIO::errorType err = img.load(imageFilename, &imgFormat);

  if (err != ImageIO::OK) 
  {
    printf("Loading texture from %s failed.\n", imageFilename);
    return -1;
  }

  // Check that the number of bytes is a multiple of 4.
  if (img.getWidth() * img.getBytesPerPixel() % 4) 
  {
    printf("Error (%s): The width*numChannels in the loaded image must be a multiple of 4.\n", imageFilename);
    return -1;
  }

  // Allocate space for an array of pixels.
  int width = img.getWidth();
  int height = img.getHeight();
  unsigned char * pixelsRGBA = new unsigned char[4 * width * height]; // we will use 4 bytes per pixel, i.e., RGBA

  // Fill the pixelsRGBA array with the image pixels.
  memset(pixelsRGBA, 0, 4 * width * height); // set all bytes to 0
  for (int h = 0; h < height; h++)
    for (int w = 0; w < width; w++) 
    {
      // assign some default byte values (for the case where img.getBytesPerPixel() < 4)
      pixelsRGBA[4 * (h * width + w) + 0] = 0; // red
      pixelsRGBA[4 * (h * width + w) + 1] = 0; // green
      pixelsRGBA[4 * (h * width + w) + 2] = 0; // blue
      pixelsRGBA[4 * (h * width + w) + 3] = 255; // alpha channel; fully opaque

      // set the RGBA channels, based on the loaded image
      int numChannels = img.getBytesPerPixel();
      for (int c = 0; c < numChannels; c++) // only set as many channels as are available in the loaded image; the rest get the default value
        pixelsRGBA[4 * (h * width + w) + c] = img.getPixel(w, h, c);
    }

  // Bind the texture.
  glBindTexture(GL_TEXTURE_2D, textureHandle);

  // Initialize the texture.
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixelsRGBA);

  // Generate the mipmaps for this texture.
  glGenerateMipmap(GL_TEXTURE_2D);

  // Set the texture parameters.
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  // Query support for anisotropic texture filtering.
  GLfloat fLargest;
  glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &fLargest);
  printf("Max available anisotropic samples: %f\n", fLargest);
  // Set anisotropic texture filtering.
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 0.5f * fLargest);

  // Query for any errors.
  GLenum errCode = glGetError();
  if (errCode != 0) 
  {
    printf("Texture initialization error. Error code: %d.\n", errCode);
    return -1;
  }
  
  // De-allocate the pixel array -- it is no longer needed.
  delete [] pixelsRGBA;

  return 0;
}

void getTextures(){
  glGenTextures(1,&groundHandle);
  int code = initTexture("texturejpgs/cringe.jpg",groundHandle);
  if (code != 0){
    printf("Error loading ground shader.");
    exit(EXIT_FAILURE);
  }
  texturePP->Bind();
  grounduvVBO = new VBO(grounduv.size() / 2,2,&grounduv[0],GL_STATIC_DRAW);
  groundVBO = new VBO(ground.size() / 3,3,&ground[0],GL_STATIC_DRAW);
  groundVAO = new VAO();
  groundVAO->ConnectPipelineProgramAndVBOAndShaderVariable(texturePP,groundVBO,"position");
  groundVAO->ConnectPipelineProgramAndVBOAndShaderVariable(texturePP,grounduvVBO,"texCoord");
  pipelineProgram->Bind();
}
// starter end

// Write a screenshot to the specified filename.
void saveScreenshot(const char * filename)
{
  unsigned char * screenshotData = new unsigned char[windowWidth * windowHeight * 3];
  glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, screenshotData);

  ImageIO screenshotImg(windowWidth, windowHeight, 3, screenshotData);

  if (screenshotImg.save(filename, ImageIO::FORMAT_JPEG) == ImageIO::OK)
    std::cout << "File " << filename << " saved successfully." << endl;
  else std::cout << "Failed to save file " << filename << '.' << endl;

  delete [] screenshotData;
}

void idleFunc()
{
  // Do some stuff... 
  // For example, here, you can save the screenshots to disk (to make the animation).
  // Notify GLUT that it should call displayFunc.
  if (ride){
      if (ind < splineP.size()) ind += 3; // since x,y,z stored in 1D fashion in splineP, tanP, binoP
      else ind = 0;
  }
  if (record){
    frame15++;
    if ((frame15 == 5) && (sc < 1000)){
      frame15 = 0;
      string fn = to_string(sc);
      if (sc < 10){
        fn = "000" + fn;
      } else if (sc < 100){
        fn = "00" + fn;
      } else fn = "0" + fn;
      fn = fn + ".jpg";
      saveScreenshot(fn.c_str());
      std::cout << "Saved Screenshot: " << fn << endl;
      sc++;
    }
  }
  glutPostRedisplay();
}

void reshapeFunc(int w, int h)
{
  glViewport(0, 0, w, h);

  // When the window has been resized, we need to re-set our projection matrix.
  matrix.SetMatrixMode(OpenGLMatrix::Projection);
  matrix.LoadIdentity();
  // You need to be careful about setting the zNear and zFar. 
  // Anything closer than zNear, or further than zFar, will be culled.
  const float zNear = 0.1f;
  const float zFar = 10000.0f;
  const float humanFieldOfView = 60.0f;
  matrix.Perspective(humanFieldOfView, 1.0f * w / h, zNear, zFar);
  // matrix.SetMatrixMode(OpenGLMatrix::ModelView);
}

void mouseMotionDragFunc(int x, int y)
{
  // Mouse has moved, and one of the mouse buttons is pressed (dragging).

  // the change in mouse position since the last invocation of this function
  int mousePosDelta[2] = { x - mousePos[0], y - mousePos[1] };

  switch (controlState)
  {
    // translate the terrain
    case TRANSLATE:
      if (leftMouseButton)
      {
        // control x,y translation via the left mouse button
        terrainTranslate[0] += mousePosDelta[0] * 0.01f;
        terrainTranslate[1] -= mousePosDelta[1] * 0.01f;
      }
      if (middleMouseButton)
      {
        // control z translation via the middle mouse button
        terrainTranslate[2] += mousePosDelta[1] * 0.01f;
      }
      break;

    // rotate the terrain
    case ROTATE:
      if (leftMouseButton)
      {
        // control x,y rotation via the left mouse button
        terrainRotate[0] += mousePosDelta[1];
        terrainRotate[1] += mousePosDelta[0];
      }
      if (middleMouseButton)
      {
        // control z rotation via the middle mouse button
        terrainRotate[2] += mousePosDelta[1];
      }
      break;

    // scale the terrain
    case SCALE:
      if (leftMouseButton)
      {
        // control x,y scaling via the left mouse button
        terrainScale[0] *= 1.0f + mousePosDelta[0] * 0.01f;
        terrainScale[1] *= 1.0f - mousePosDelta[1] * 0.01f;
      }
      if (middleMouseButton)
      {
        // control z scaling via the middle mouse button
        terrainScale[2] *= 1.0f - mousePosDelta[1] * 0.01f;
      }
      break;
  }

  // store the new mouse position
  mousePos[0] = x;
  mousePos[1] = y;
}

void mouseMotionFunc(int x, int y)
{
  // Mouse has moved.
  // Store the new mouse position.
  mousePos[0] = x;
  mousePos[1] = y;
}

void mouseButtonFunc(int button, int state, int x, int y)
{
  // A mouse button has has been pressed or depressed.

  // Keep track of the mouse button state, in leftMouseButton, middleMouseButton, rightMouseButton variables.
  switch (button)
  {
    case GLUT_LEFT_BUTTON:
      leftMouseButton = (state == GLUT_DOWN);
    break;

    case GLUT_MIDDLE_BUTTON:
      middleMouseButton = (state == GLUT_DOWN);
    break;

    case GLUT_RIGHT_BUTTON:
      rightMouseButton = (state == GLUT_DOWN);
    break;
  }

  // Keep track of whether CTRL and SHIFT keys are pressed.
  switch (glutGetModifiers())
  {
    // case GLUT_ACTIVE_CTRL:
    //   controlState = TRANSLATE;
    // break;

    case GLUT_ACTIVE_SHIFT:
      controlState = SCALE;
    break;

    // If CTRL and SHIFT are not pressed, we are in rotate mode.
    default:
      controlState = ROTATE;
    break;
  }

  // Store the new mouse position.
  mousePos[0] = x;
  mousePos[1] = y;
}

void keyboardFunc(unsigned char key, int x, int y)
{
  switch (key)
  {
    case 27: // ESC key
      exit(0); // exit the program
    break;

    case ' ':
      std::cout << "You pressed the spacebar." << endl;
    break;

    case 'x':
      // Take a screenshot.
      saveScreenshot("screenshot.jpg");
    break;

    case 't': // since GLUT_ACTIVE_CTRL doesn't work on mac, use t for translate
      controlState = TRANSLATE;
    break;

    case 'm':
      if (middleMouseButton == 1) middleMouseButton = 0;
      else middleMouseButton = 1;
    break;

    case 'w':
      ride = !ride;
      // ind = 0;
      std::cout << "toggle ride on/off" << endl;
    break;

    case 'r':
      record = !record;
    break;

    case 'a': // find default lookAt()
      add = !add;
    break;

    case '1': 
      if (add) eyeD[0] += change;
      else eyeD[0] -= change;
      std::cout << "eyeD[0],x: " << eyeD[0] << endl;
    break;

    case '2':
      if (add) eyeD[1] += change;
      else eyeD[1] -= change;
      std::cout << "eyeD[1],y: " << eyeD[1] << endl;
    break;

    case '3':
      if (add) eyeD[2] += change;
      else eyeD[2] -= change;
      std::cout << "eyeD[2],z: " << eyeD[2] << endl;
    break;

    case '4':
      if (add) centerD[0] += change;
      else centerD[0] -= change;
      std::cout << "centerD[0],x: " << centerD[0] << endl;
    break;

    case '5':
      if (add) centerD[1] += change;
      else centerD[0] -= change;
      std::cout << "centerD[1],y: " << centerD[1] << endl;
    break;

    case '6':
      if (add) centerD[2] += change;
      else centerD[2] -= change;
      std::cout << "centerD[2],z: " << centerD[2] << endl;
    break;

    case 'd':
      std::cout << "eyeD[]: ";
      for (int i=0;i<3;i++){
        std::cout << ", " << eyeD[i];
      }
      std::cout << "; centerD[]: ";
      for (int i=0;i<3;i++){
        std::cout << ", " << centerD[i];
      }
    break;
  }
}

void displayFunc()
{
  // This function performs the actual rendering.

  // First, clear the screen.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Set up the camera position, focus point, and the up vector.
  matrix.SetMatrixMode(OpenGLMatrix::ModelView);
  matrix.LoadIdentity();

  // should bino be neg???
  float ot = 0.2f;
  glm::vec3 eye(splineP[ind]-ot*binoP[ind],splineP[ind+1]-ot*binoP[ind+1],splineP[ind+2]-ot*binoP[ind+2]); // add with color1c -> normals
  glm::vec3 center(splineP[ind]+tanP[ind]-ot*binoP[ind],splineP[ind+1]+tanP[ind+1]-ot*binoP[ind+1],splineP[ind+2]+tanP[ind+2]-ot*binoP[ind+2]);
  glm::vec3 up(-binoP[ind],-binoP[ind+1],-binoP[ind+2]);

  if (ride) matrix.LookAt(eye.x,eye.y,eye.z,center.x,center.y,center.z,up.x,up.y,up.z);
  else matrix.LookAt(eyeD[0],eyeD[1],eyeD[2],centerD[0],centerD[1],centerD[2], 0.0, 1.0, 0.0); // experimental "best" values

  // translate and scale by values from arrays
  matrix.Translate(terrainTranslate[0],terrainTranslate[1],terrainTranslate[2]);
  matrix.Scale(terrainScale[0],terrainScale[1],terrainScale[2]);
  // rotate by value from arrays in each axis: x, y, z
  matrix.Rotate(terrainRotate[0],1,0,0);
  matrix.Rotate(terrainRotate[1],0,1,0);
  matrix.Rotate(terrainRotate[2],0,0,1);
  

  // Read the current modelview and projection matrices from our helper class.
  // The matrices are only read here; nothing is actually communicated to OpenGL yet.
  float modelViewMatrix[16];
  matrix.SetMatrixMode(OpenGLMatrix::ModelView);
  matrix.GetMatrix(modelViewMatrix);

  float projectionMatrix[16];
  matrix.SetMatrixMode(OpenGLMatrix::Projection);
  matrix.GetMatrix(projectionMatrix);

  double lightDirection[4] = { 0, 1, 0, 0 }; // the “Sun” at noon
  float viewLightDirection[3]; // light direction in the view space
  matrix.SetMatrixMode(OpenGLMatrix::ModelView);
  matrix.GetMatrix(modelViewMatrix); // read the view matrix
  double view[16];
  for (int i=0;i<16;i++) {
    view[i] = static_cast<double>(modelViewMatrix[i]);
  }
  double tempresult[3];
  // viewLightDirection = (view * float4(lightDirection, 0.0)).xyz;
  MultiplyMatrices(4,4,1,view,lightDirection,tempresult);
  viewLightDirection[0] = static_cast<float>(tempresult[0]);
  viewLightDirection[1] = static_cast<float>(tempresult[1]);
  viewLightDirection[2] = static_cast<float>(tempresult[2]);
  pipelineProgram->SetUniformVariable3fv("viewLightDirection",viewLightDirection);

  float n[16];
  matrix.SetMatrixMode(OpenGLMatrix::ModelView);
  matrix.GetNormalMatrix(n); // get normal matrix
  pipelineProgram->SetUniformVariableMatrix4fv("normalMatrix", GL_FALSE, n);

  float ka[4] = {0.1, 0.1, 0.1, 0.0};
  float kd[4] = {0.5, 0.5, 0.5, 0.0};
  float ks[4] = {0.1, 0.1, 0.1, 0.0};
  float L[4] = {1.5, 1.5, 1.5, 1.5};

  pipelineProgram->SetUniformVariable4fv("ka",ka);
  pipelineProgram->SetUniformVariable4fv("kd",kd);
  pipelineProgram->SetUniformVariable4fv("ks",ks);

  pipelineProgram->SetUniformVariable4fv("La",L);
  pipelineProgram->SetUniformVariable4fv("Ld",L);
  pipelineProgram->SetUniformVariable4fv("Ls",L);

  pipelineProgram->SetUniformVariablef("alpha",0.5);


  // pass phong values

  // Upload the modelview and projection matrices to the GPU. Note that these are "uniform" variables.
  // Important: these matrices must be uploaded to *all* pipeline programs used.
  // In hw1, there is only one pipeline program, but in hw2 there will be several of them.
  // In such a case, you must separately upload to *each* pipeline program.
  // Important: do not make a typo in the variable name below; otherwise, the program will malfunction.
  pipelineProgram->SetUniformVariableMatrix4fv("modelViewMatrix", GL_FALSE, modelViewMatrix);
  pipelineProgram->SetUniformVariableMatrix4fv("projectionMatrix", GL_FALSE, projectionMatrix);

  // Execute the rendering.
  // Bind the VAO that we want to render. Remember, one object = one VAO. 
  trackVAO->Bind();
  glDrawArrays(GL_TRIANGLES,0,numVertices);

  texturePP->Bind();
  texturePP->SetUniformVariableMatrix4fv("modelViewMatrix", GL_FALSE, modelViewMatrix);
  texturePP->SetUniformVariableMatrix4fv("projectionMatrix", GL_FALSE, projectionMatrix);
  setTextureUnit(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D,groundHandle);
  groundVAO->Bind();
  glDrawArrays(GL_TRIANGLES,0,ground.size() / 3);


  pipelineProgram->Bind();
  // Swap the double-buffers.
  glutSwapBuffers();
}

void initScene(int argc, char *argv[])
{
  // Set the background color.
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black color.

  // Enable z-buffering (i.e., hidden surface removal using the z-buffer algorithm).
  glEnable(GL_DEPTH_TEST);

  // Create a pipeline program. This operation must be performed BEFORE we initialize any VAOs.
  // A pipeline program contains our shaders. Different pipeline programs may contain different shaders.
  // In this homework, we only have one set of shaders, and therefore, there is only one pipeline program.
  // In hw2, we will need to shade different objects with different shaders, and therefore, we will have
  // several pipeline programs (e.g., one for the rails, one for the ground/sky, etc.).
  // Load and set up the pipeline program, including its shaders.
  // pipelineProgram = new PipelineProgram(); // Load and set up the pipeline program, including its shaders.
  // if (pipelineProgram->BuildShadersFromFiles(shaderBasePath, "vertexShader.glsl", "fragmentShader.glsl") != 0)
  // {
  //   std::cout << "Failed to build the pipeline program." << endl;
  //   throw 1;
  // } 
  // std::cout << "Successfully built the pipeline program." << endl;

  texturePP = new PipelineProgram();
  if (texturePP->BuildShadersFromFiles(shaderBasePath, "textureShader.glsl", "textureFragmentShader.glsl") != 0)
  {
    std::cout << "Failed to build the texture pipeline program." << endl;
    throw 1;
  } 
  std::cout << "Successfully built the texture pipeline program." << endl;

  pipelineProgram = new PipelineProgram();
  if (pipelineProgram->BuildShadersFromFiles(shaderBasePath, "phongShader.glsl", "phongFragmentShader.glsl") != 0)
  {
    std::cout << "Failed to build the phong pipeline program." << endl;
    throw 1;
  } 
  std::cout << "Successfully built the phong pipeline program." << endl;
    
  // Bind the pipeline program that we just created. 
  // The purpose of binding a pipeline program is to activate the shaders that it contains, i.e.,
  // any object rendered from that point on, will use those shaders.
  // When the application starts, no pipeline program is bound, which means that rendering is not set up.
  // So, at some point (such as below), we need to bind a pipeline program.
  // From that point on, exactly one pipeline program is bound at any moment of time.
  pipelineProgram->Bind();

  double temp[12] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
  double result[3] = {0.0,0.0,0.0};
  double first[3] = {0.0,0.0,0.0};
  double holder[3] = {0.0,0.0,0.0};
  // calculate points here
  for (int i=0;i<spline.numControlPoints-3;i++){
    // 3x4 col-major control matrix, transposed to 4x3 for MM
    double control[12] = {spline.points[i].x,spline.points[i+1].x,spline.points[i+2].x,spline.points[i+3].x,
                          spline.points[i].y,spline.points[i+1].y,spline.points[i+2].y,spline.points[i+3].y,
                          spline.points[i].z,spline.points[i+1].z,spline.points[i+2].z,spline.points[i+3].z
    };
    MultiplyMatrices(4,4,3,basis,control,temp);
    for (float u=0.000f;u<=1.000f;u+=0.001f){
      double umatrix[4] = {u*u*u,u*u,u,1};
      double dumatrix[4] = {3*(u*u),2*u,1,0};
      MultiplyMatrices(1,4,3,umatrix,temp,result);
      splineP.push_back(result[0]);
      splineP.push_back(result[1]);
      splineP.push_back(result[2]);
      splineC.push_back(1.0);
      splineC.push_back(1.0);
      splineC.push_back(1.0);
      splineC.push_back(1.0);
      MultiplyMatrices(1,4,3,dumatrix,temp,result);
      unit(result);
      tanP.push_back(result[0]);
      tanP.push_back(result[1]);
      tanP.push_back(result[2]);
      if ((i == 0) && (u == 0.000f)){
        calcUpVec(result[0],result[1],result[2],true);
      } else{
        calcUpVec(result[0],result[1],result[2],false);
        makeTracks();
      }
    }
  }
  makeGround();
  getTextures();
  // number of vertices for spline
  numVertices = track1.size() / 3;
  track1VBO = new VBO(numVertices,3,&track1[0],GL_STATIC_DRAW);
  track1cVBO = new VBO(numVertices,3,&track1c[0],GL_STATIC_DRAW);
  // track1cVBO = new VBO(numVertices,4,&track1c[0],GL_STATIC_DRAW);

  trackVAO = new VAO();
  trackVAO->ConnectPipelineProgramAndVBOAndShaderVariable(pipelineProgram, track1VBO, "position");
  // trackVAO->ConnectPipelineProgramAndVBOAndShaderVariable(pipelineProgram, track1cVBO, "color");
  trackVAO->ConnectPipelineProgramAndVBOAndShaderVariable(pipelineProgram, track1cVBO, "normal");
  // Check for any OpenGL errors.
  std::cout << "GL error status is: " << glGetError() << std::endl;
}

int main(int argc, char *argv[])
{
  if (argc < 2)
  {  
    printf ("Usage: %s <spline file>\n", argv[0]);
    exit(0);
  }
  // Load spline from the provided filename.
  loadSpline(argv[1]);
  printf("Loaded spline with %d control point(s).\n", spline.numControlPoints);

  std::cout << "Initializing GLUT..." << endl;
  glutInit(&argc,argv);

  std::cout << "Initializing OpenGL..." << endl;

  #ifdef __APPLE__
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
  #else
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
  #endif

  glutInitWindowSize(windowWidth, windowHeight);
  glutInitWindowPosition(0, 0);  
  glutCreateWindow(windowTitle);

  std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << endl;
  std::cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << endl;
  std::cout << "Shading Language Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;

  #ifdef __APPLE__
    // This is needed on recent Mac OS X versions to correctly display the window.
    glutReshapeWindow(windowWidth - 1, windowHeight - 1);
  #endif

  // Tells GLUT to use a particular display function to redraw.
  glutDisplayFunc(displayFunc);
  // Perform animation inside idleFunc.
  glutIdleFunc(idleFunc);
  // callback for mouse drags
  glutMotionFunc(mouseMotionDragFunc);
  // callback for idle mouse movement
  glutPassiveMotionFunc(mouseMotionFunc);
  // callback for mouse button changes
  glutMouseFunc(mouseButtonFunc);
  // callback for resizing the window
  glutReshapeFunc(reshapeFunc);
  // callback for pressing the keys on the keyboard
  glutKeyboardFunc(keyboardFunc);

  // init glew
  #ifdef __APPLE__
    // nothing is needed on Apple
  #else
    // Windows, Linux
    GLint result = glewInit();
    if (result != GLEW_OK)
    {
      cout << "error: " << glewGetErrorString(result) << endl;
      exit(EXIT_FAILURE);
    }
  #endif

  // Perform the initialization.
  initScene(argc, argv);

  // Sink forever into the GLUT loop.
  glutMainLoop();
}

