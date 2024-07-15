
import whisper
import zhconv
model = whisper.load_model("medium", download_root='/home/scc/Downloads/Models')
# Mandarin zh chinese-(台湾)
result = model.transcribe(
    "/home/scc/sccWork/myGitHub/PaperWithCode-burningGPU/myPP/putong2putong/新录音.m4a", 
    language="Mandarin",
    condition_on_previous_text=True,
    word_timestamps=True,
    verbose=True
)
out_txt = '/home/scc/Downloads/Models/text_out/result.txt'
print(result['text'])


shell_ = """
whisper  ./新录音.m4a --model medium --model_dir '/home/scc/Downloads/Models' --language 'Mandarin' --task transcribe \
    --output_dir '/home/scc/Downloads/Models/text_out' --output_format txt \
    --condition_on_previous_text True \
    --word_timestamps True

"""
 
import zhconv
simplified_chinese_text = zhconv.convert(result['text'], 'zh-cn') 
simplified_chinese_text[:10]

with open(out_txt, 'w', encoding='gbk') as f:
    f.write(simplified_chinese_text)


# ==========================================================================================================
# python3 
# https://www.modelscope.cn/models/iic/Whisper-large-v3/files
# pip install --use-pep517 modelscope[audio] -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='iic/Whisper-large-v3', 
    model_revision="v2.0.5"
)

rec_result = inference_pipeline(input='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav', language=None)
print(rec_result)




