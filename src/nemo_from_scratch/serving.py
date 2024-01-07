import bentoml

runner = 

svc = bentoml.Service(name="voice activity detection.", runner= [runner])

@svc.api(input=, output=)
async def VAD(audio: )