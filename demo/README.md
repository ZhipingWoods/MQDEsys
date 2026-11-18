# MQDEsys interactive demo

The static showcase presents screenshots and recorded walkthroughs. This Gradio application provides an interactive interface to a running MQDEsys instance through its REST API.

```bash
cd /path/to/MQDEsys
pip install -r demo/requirements-gradio.txt
MQDE_API=http://127.0.0.1:18081 python demo/gradio_app.py
# open http://127.0.0.1:7860/
```

The `MQDE_API` environment variable points to the MQDEsys server. The default is `http://127.0.0.1:18081`.

## Tabs

| Tab | Function |
|---|---|
| Overview | GPU pool, probe devices, inference backends and recent jobs |
| Resource estimate | Model × recipe × hardware; peak memory, concurrency and latency floor |
| Deployment fit | Rank local models or the Hugging Face catalogue |
| Submit job | Build a job specification, preview it, then submit |
| Model search / download | Hugging Face and ModelScope search, weight groups and download status |
| Benchmark catalogue | Offline, native (`datasets`) and EvalScope datasets |

The demo uses the same REST API as the web workspace and does not maintain separate state.

## Remote access

If the services run on a remote host, forward both ports:

```bash
ssh -N -L 7860:127.0.0.1:7860 -L 18081:127.0.0.1:18081 <user>@<server>
```

## Security

The Gradio application is intended for local evaluation. Neither it nor the MQDEsys server enables authentication; bind both to loopback and use an SSH tunnel for remote access.
