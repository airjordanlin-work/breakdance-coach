## May 19
Built the normalization function. Chose hip-center as origin 
because it's the most stable point on the body during movement. 
Torso scale instead of height because height changes when someone crouches. fastdtw over regular DTW because O(n) vs O(n²) matters at 30fps.
Additionally handle occlusion(body part hidden from camera)MediaPipe predicts occluded landmarks but flags them with low visibility scores, so you threshold on visibility before feeding frames into the DTW buffer. Only high-confidence frames enter your comparison pipeline


##MAY 24
Forgot to update but I added a front end design and implemented the camera/ability to toggle

Improving skeleton model: 
Backend sends ~2KB JSON per frame instead of ~50KB JPEG — roughly 25x less data over the WebSocket
Frontend owns the rendering — smoother, no compression artifacts
Ghost skeleton is now drawn in React canvas space so it's perfectly crisp
Guidance is now joint-specific instead of just "step back"