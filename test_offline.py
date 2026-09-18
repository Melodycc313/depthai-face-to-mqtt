"""No camera opened. Run with: python test_offline.py"""
import sys
import tempfile
from pathlib import Path
import numpy as np
import face_login_service as m

orig = m.DATA_DIR
try:
    with tempfile.TemporaryDirectory() as d:
        m.DATA_DIR = Path(d) / 'profiles'
        c = m.FaceController()
        assert c.status()['state'] == 'idle'
        assert c.handle({'command':'register','user_id':'alice'})['ok'] is False
        assert c.handle({'command':'register','consent':True,'user_id':'../escape'})['ok'] is False
        assert c.handle({'command':'login','consent':True})['ok'] is False
        vector = np.zeros(128, dtype=np.float32); vector[0] = 1
        c.state='running'; c.operation='register'; c.enroll_user='alice'; c.last_sample_at=0; c.deadline=m.monotonic()+100
        m.SAMPLE_GAP_SECONDS=0
        for n in range(1,13): c._process_embedding(n,vector)
        assert c.status()['state']=='registered' and c.status()['user_id']=='alice'
        assert len(m.load_profiles()['alice'])==12
        assert c.handle({'command':'register','consent':True,'user_id':'alice'})['ok'] is False
        assert list(m.DATA_DIR.glob('*.jpg')) == []
        c2=m.FaceController(); c2.state='running'; c2.operation='login'; c2.profiles=m.load_profiles(); c2.deadline=m.monotonic()+100
        for n in range(1,6): c2._process_embedding(n,vector)
        assert c2.status()['state']=='matched_experimental' and c2.status()['authenticated']
        assert c2.handle({'command':'guest'})['state']=='guest'
        assert c2.status()['authenticated'] is False and c2.status()['user_id'] is None
        print('OFFLINE PASS: consent, ID validation, no overwrite, 12 samples, matching, guest reset')
finally:
    m.DATA_DIR=orig
