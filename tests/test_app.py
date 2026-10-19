from pathlib import Path
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'app.py'

def test_demo_all_pages_and_session_isolation():
    a=AppTest.from_file(str(APP),default_timeout=30).run()
    assert not a.exception
    a.button(key='demo').click().run()
    assert not a.exception and len(a.session_state['table'])==36
    for page in ['Organizar','Explorar','Projeto','Importar']:
        a.radio(key='page').set_value(page).run()
        assert not a.exception, [e.message for e in a.exception]
    b=AppTest.from_file(str(APP),default_timeout=30).run()
    assert 'table' not in b.session_state
    b.radio(key='page').set_value('Projeto').run()
    assert not b.exception
    assert len(a.session_state['table'])==36

def test_upload_csv_and_restore_project():
    from core.data import demo,save_project
    a=AppTest.from_file(str(APP),default_timeout=30).run()
    a.file_uploader(key='table_upload').upload('dados.csv',b'900;905\n1,2;3,4\n2,3;4,5').run()
    next(b for b in a.button if b.label=='Importar tabela').click().run()
    assert not a.exception and a.session_state['table'].shape==(2,2)
    a.radio(key='page').set_value('Explorar').run()
    assert not a.exception
    df,cfg=demo();cfg['excluded']=[2]
    a.radio(key='page').set_value('Projeto').run()
    a.file_uploader(key='project_upload').upload('exemplo.zip',save_project(df,cfg,'Restaurado')).run()
    next(b for b in a.button if b.label=='Abrir projeto').click().run()
    assert not a.exception
    assert a.session_state['project_name']=='Restaurado'
    assert a.session_state['config']['excluded']==[2]
    a.radio(key='page').set_value('Explorar').run()
    assert not a.exception and a.metric[0].value=='35'
