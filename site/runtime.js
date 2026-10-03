export async function load(onProgress) {
  const [{Schemer},ort,{PreTrainedTokenizer,env}] = await Promise.all([
    import('./engine/engine.js'),
    import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.bundle.min.mjs'),
    import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0'),
  ]);
  env.allowRemoteModels=false;
  ort.env.wasm.numThreads=1;
  ort.env.wasm.proxy=true;
  ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
  const base='https://huggingface.co/spaces/desert-ant-labs/schemer-demo/resolve/7676626a50b9d8549967e9cf980e4d9c9e717fc1/model';
  return Schemer.load({ort,PreTrainedTokenizer,modelBase:base,specBase:base+'/harness',executionProviders:['wasm'],lowMemory:true,maxLength:512,onProgress:({loaded,total})=>onProgress(total?loaded/total:0)});
}
