"""Serialization and security checks for portable trace replay."""

import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from rtga.viewer import write_viewer


def fixture_trace():
    """A deliberately tiny synthetic rendering fixture, never a result claim."""
    return {
        "version": 1,
        "metadata": {"title": "Synthetic viewer test fixture", "method": "fixture", "seed": 0},
        "environment": {"width": 1, "height": 1, "wall_x": .6, "door_y": [.4, .6],
                        "switch": [.25, .7], "noise_source": [.8, .2], "variant": "noise"},
        "frames": [
            {"step": 0, "state": [.1, .2, .01, 0, 0, 0], "action": 2,
             "objective": .2, "plans": [[2, 3, 1], [1, 0, 4]],
             "paths": [[[.1, .2], [.3, .3], [.4, .5]]],
             "selected_path": [[.1, .2], [.3, .3], [.4, .5]],
             "prediction_error": None, "disagreement": .03, "latency_ms": 2.4},
            {"step": 1, "state": [.2, .22, .02, .01, 1, .4], "action": 3,
             "objective": .1, "plans": [[3, 3, 2], [2, 4, 0]],
             "prediction_error": .01, "disagreement": None, "latency_ms": 3.1},
        ],
    }


class ViewerTests(unittest.TestCase):
    def test_embedded_trace_round_trips_and_creates_parent(self):
        trace = fixture_trace()
        with tempfile.TemporaryDirectory() as directory:
            path = write_viewer(trace, Path(directory) / "nested" / "trace.html")
            document = path.read_text()
        match = re.search(r'<script type="application/json" id="trace-data">(.*?)</script>', document)
        self.assertIsNotNone(match)
        self.assertEqual(json.loads(match.group(1)), trace)
        self.assertNotIn('<script src=', document)
        self.assertNotIn('__TRACE_', document)

    def test_script_breakout_and_title_are_escaped_without_losing_data(self):
        trace = fixture_trace()
        malicious = '</script><script>alert("x")</script><img src=x onerror=alert(1)> & \u2028'
        trace["metadata"]["title"] = malicious
        trace["frames"][0]["metrics"] = {"description": malicious}
        with tempfile.TemporaryDirectory() as directory:
            document = write_viewer(trace, Path(directory) / "trace.html").read_text()
        self.assertNotIn('<script>alert(', document)
        self.assertNotIn('<img src=x', document)
        self.assertIn('&lt;/script&gt;', document)
        payload = re.search(r'id="trace-data">(.*?)</script>', document).group(1)
        self.assertNotIn('<', payload)
        self.assertEqual(json.loads(payload), trace)

    def test_empty_trace_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_viewer({"frames": []}, Path(directory) / "empty.html")
            self.assertIn('No frames were recorded', path.read_text())

    def test_template_tokens_inside_user_data_are_preserved(self):
        trace = fixture_trace()
        trace["metadata"]["title"] = "__TRACE_PAYLOAD__ __TRACE_TITLE__"
        with tempfile.TemporaryDirectory() as directory:
            document = write_viewer(trace, Path(directory) / "trace.html").read_text()
        payload = re.search(r'id="trace-data">(.*?)</script>', document).group(1)
        self.assertEqual(json.loads(payload), trace)
        self.assertIn('<title>__TRACE_PAYLOAD__ __TRACE_TITLE__</title>', document)

    def test_missing_frames_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'frames list'):
            write_viewer({}, '/unused/path.html')

    def test_nonfinite_data_is_not_silently_encoded_as_invalid_json(self):
        trace = fixture_trace()
        trace["frames"][0]["prediction_error"] = float('nan')
        with self.assertRaises(ValueError):
            write_viewer(trace, '/unused/path.html')

    @unittest.skipUnless(shutil.which("node"), "Node is needed for JavaScript execution checks")
    def test_controls_and_optional_data_in_mock_dom(self):
        """Execute generated JS; this checks behavior, not browser visual layout."""
        with tempfile.TemporaryDirectory() as directory:
            open_trace = fixture_trace()
            open_trace["environment"].update(variant="open", goal=[.8, .7])
            paths = [write_viewer(fixture_trace(), Path(directory) / "trace.html"),
                     write_viewer({"frames": [{"state": [.1, .2]}]}, Path(directory) / "minimal.html"),
                     write_viewer({"frames": []}, Path(directory) / "empty.html"),
                     write_viewer(open_trace, Path(directory) / "open.html")]
            completed = subprocess.run([shutil.which("node"), "-e", _MOCK_DOM, *map(str, paths)],
                                       capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


_MOCK_DOM = r"""
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
for (const path of process.argv.slice(1)) for (const width of [360,1200]) {
  const source=fs.readFileSync(path,'utf8'),data=source.match(/id="trace-data">(.*?)<\/script>/s)[1];
  const script=source.match(/<script>(.*?)<\/script>/s)[1],elements=new Map(),labels=[],strokes=[];
  const c=new Proxy({}, {get:(target,key)=>target[key]||((...args)=>{if(key==='fillText')labels.push(args[0]);if(key==='stroke')strokes.push(target.strokeStyle);for(const x of args)if(typeof x==='number')assert.ok(Number.isFinite(x),key+' had nonfinite coordinates');}),set:(target,key,value)=>{target[key]=value;return true;}});
  function element(id='') {return {textContent:id==='trace-data'?data:'',value:id==='speed'?'10':'0',hidden:false,style:{setProperty(){}},handlers:{},setAttribute(){},append(){},addEventListener(name,fn){this.handlers[name]=fn;},setPointerCapture(){},getBoundingClientRect(){return {width:width<760?width-26:(width-70)/2,height:id==='timeline'?236:405,left:0};},getContext(){return c;}};}
  const document={getElementById(id){if(!elements.has(id))elements.set(id,element(id));return elements.get(id);},createElement(){return element();},createTextNode(text){return text;},addEventListener(){}};
  const sandbox={document,window:{devicePixelRatio:2},performance:{now:()=>0},requestAnimationFrame(){},ResizeObserver:class{observe(){}},console};
  vm.createContext(sandbox);vm.runInContext(script,sandbox);
  const get=id=>elements.get(id),count=JSON.parse(data).frames.length;
  if(!count){assert.equal(get('replay').hidden,true);assert.equal(get('empty').hidden,false);continue;}
  assert.equal(get('position').textContent,'1 / '+count);
  if(JSON.parse(data).environment?.variant==='open'){assert.ok(labels.includes('goal'));assert.ok(!labels.includes('switch'));assert.ok(!labels.includes('noise'));assert.ok(!strokes.includes('#a7b1bb'));assert.equal(get('switch-legend').hidden,true);assert.equal(get('noise-legend').hidden,true);}
  if(count===2){get('next').onclick();assert.equal(get('position').textContent,'2 / 2');assert.equal(get('error').textContent,'0.01');assert.equal(get('disagreement').textContent,'—');get('previous').onclick();assert.equal(get('error').textContent,'—');get('seek').oninput({target:{value:'1'}});assert.equal(get('next').disabled,true);get('play').onclick();assert.equal(get('position').textContent,'1 / 2');assert.equal(get('play').textContent,'Pause');get('play').onclick();assert.equal(get('play').textContent,'Play');}
  else{assert.equal(get('action').textContent,'—');assert.equal(get('objective').textContent,'—');assert.equal(get('latency').textContent,'—');}
}
"""


if __name__ == "__main__":
    unittest.main()
