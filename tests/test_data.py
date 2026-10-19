from copy import deepcopy
import io,json,zipfile
import numpy as np
import pandas as pd
import pytest
from core.data import *


def test_project_roundtrip():
    df,cfg=demo(); cfg['excluded']=[1,5]; cfg['notes']='Ensaio áç; unidades de Y: mg/L'
    restored,settings,name=load_project(save_project(df,cfg,'Teste'))
    pd.testing.assert_frame_equal(df,restored)
    assert cfg==settings and name=='Teste'
    a,b=build_dataset(df,cfg),build_dataset(restored,settings)
    assert len(a['X'])==34
    for k in ['X','Y']:pd.testing.assert_frame_equal(a[k],b[k])
    assert a['ids']==b['ids'] and a['groups']==b['groups']
    np.testing.assert_array_equal(a['axis'],b['axis'])


def test_brazil_csv_preserves_id_and_missing():
    df,dec=read_table('ID;900;905\n001;1,25;\n002;2,50;3,2'.encode(),'a.csv')
    cfg=default_config(df,dec);cfg['roles']['ID']='Identificação'
    data=build_dataset(df,cfg)
    assert data['ids']==['001','002'] and data['X'].iloc[0,0]==1.25
    assert pd.isna(data['X'].iloc[0,1])


def test_excel_numeric_and_sheet():
    raw=io.BytesIO()
    with pd.ExcelWriter(raw,engine='openpyxl') as writer:
        pd.DataFrame({'ID':['001','002'],'Sinal':[1.25,2.5]}).to_excel(writer,sheet_name='Dados',index=False)
    assert excel_sheets(raw.getvalue())==['Dados']
    df,dec=read_table(raw.getvalue(),'a.xlsx',sheet='Dados')
    cfg=default_config(df,dec);cfg['roles']['ID']='Identificação'
    assert build_dataset(df,cfg)['X'].iloc[0,0]==1.25


@pytest.mark.parametrize('text',['a;a\n1;2','a;b\n1;2;3',';b\n1;2'])
def test_bad_csv(text):
    with pytest.raises(ValueError):read_table(text.encode(),'a.csv')


def test_bad_numeric_duplicate_ids_and_exclusions():
    df,cfg=demo()
    broken=df.copy();broken.iloc[0,5]='abc'
    with pytest.raises(ValueError):validate_config(broken,cfg)
    broken=df.copy();broken.iloc[0,0]=broken.iloc[1,0]
    with pytest.raises(ValueError):validate_config(broken,cfg)
    cfg['excluded']=list(range(len(df)))
    with pytest.raises(ValueError):validate_config(df,cfg)


def test_extra_zip_member_rejected():
    df,cfg=demo();raw=save_project(df,cfg,'Teste')
    buf=io.BytesIO(raw)
    with zipfile.ZipFile(buf,'a') as z:z.writestr('../unwanted','x')
    with pytest.raises(ValueError):load_project(buf.getvalue())


def test_corrupt_schema_rejected():
    df,cfg=demo();raw=save_project(df,cfg,'Teste')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:p=json.loads(z.read('project.json'))
    p['schema']=100
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:z.writestr('project.json',json.dumps(p))
    with pytest.raises(ValueError):load_project(out.getvalue())
