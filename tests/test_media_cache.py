import hashlib
import shutil
import subprocess

import pytest

from shorts_pipeline import media


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg is required')
def test_restored_source_cache_preserves_prepared_excerpt_without_reencoding(tmp_path, monkeypatch):
    source=tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-threads','1','-filter_threads','1',
                    '-f','lavfi','-i','testsrc2=size=160x160:rate=30:duration=2',
                    '-c:v','libx264','-threads','1',str(source)],check=True,timeout=30)
    first=tmp_path/'first-run'
    prepared=media.excerpt(source,first,[.25,1.25])
    checksum=hashlib.sha256(prepared.read_bytes()).hexdigest()
    assert prepared.parent==first/'sources'/'excerpts'
    second=tmp_path/'fresh-runner'
    shutil.copytree(first/'sources',second/'sources')
    original_run=media.run
    def no_encoder(args, **kwargs):
        if args[0]=='ffmpeg':
            raise AssertionError('A cache miss would need the encoder')
        return original_run(args,**kwargs)
    monkeypatch.setattr(media,'run',no_encoder)
    restored=media.excerpt(source,second,[.25,1.25])
    assert hashlib.sha256(restored.read_bytes()).hexdigest()==checksum
    assert media.probe(restored)['duration']==pytest.approx(1,abs=.05)
    with pytest.raises(AssertionError,match='cache miss'):
        media.excerpt(source,second,[.5,1.5])
    # A changed source must not reuse a prior source's prepared clip.
    with source.open('ab') as handle:
        handle.write(b'new-source-revision')
    with pytest.raises(AssertionError,match='cache miss'):
        media.excerpt(source,second,[.25,1.25])
