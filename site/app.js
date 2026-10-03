const $=id=>document.getElementById(id);
const sample='Contact Maria Rossi at maria.rossi@example.com. She ordered 3 blue notebooks for 12.50 euros each.';
const schema={name:{type:'string',describe:'Full name of the person'},email:{type:'string',describe:'Email address'},quantity:{type:'int',describe:'Number of notebooks ordered'},unit_price:{type:'float',describe:'Price of each notebook in euros'}};
$('controls').innerHTML='<label for="input">Passage</label><textarea id="input" maxlength="12000"></textarea><label for="schema">Extraction schema (JSON)</label><textarea id="schema" maxlength="12000" style="min-height:220px"></textarea><div class="actions"><button id="run">Extract structured data</button><button id="sample" class="secondary">Reset example</button></div><p class="hint">About 200 MB on first use. Supported field types include string, int, float, bool, label, datetime and duration. Each field needs a type and a describe prompt. The input is limited to 512 model tokens in this demo; use short passages. Extracted values may be wrong: verify them before use.</p>';
function reset(){$('input').value=sample;$('schema').value=JSON.stringify(schema,null,2);}
reset();$('sample').onclick=reset;
let model;
$('run').onclick=async()=>{
  $('run').disabled=$('sample').disabled=true;$('output').replaceChildren();$('output').dataset.state='loading';
  try{
    const text=$('input').value.trim();if(!text)throw Error('Enter a passage.');
    const fields=JSON.parse($('schema').value);if(!fields||Array.isArray(fields)||typeof fields!=='object'||!Object.keys(fields).length)throw Error('Schema must be a nonempty JSON object.');
    if(Object.keys(fields).length>20)throw Error('Please use no more than 20 fields.');
    $('status').textContent='Loading original Schemer models...';$('progress').hidden=false;
    model??=import('./runtime.js').then(({load})=>load(f=>{$('progress').value=f;$('status').textContent=`Loading models: ${(f*100).toFixed(0)}%`;})).catch(e=>{model=null;throw e;});
    const engine=await model;$('progress').hidden=true;
    const result=await engine.extract(text,fields,undefined,(field,i,n)=>{$('status').textContent=`Extracting ${field} (${i+1}/${n})`;});
    const pre=document.createElement('pre');pre.textContent=JSON.stringify(result,null,2);$('output').append(pre);
    $('output').dataset.state='success';$('status').textContent='Extraction complete. Review the result.';
  }catch(e){$('output').textContent=e.message??String(e);$('output').dataset.state='error';$('status').textContent='Could not complete. You can retry.';}
  finally{$('progress').hidden=true;$('run').disabled=$('sample').disabled=false;}
};
