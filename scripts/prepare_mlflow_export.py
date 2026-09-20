import os
import shutil
import sqlite3
import tarfile

def prepare_mlflow():
    export_dir = os.path.join(os.getcwd(), 'mlflow_export')
    os.makedirs(export_dir, exist_ok=True)
    
    # Copy db
    dst_db = os.path.join(export_dir, 'mlflow.db')
    shutil.copyfile('mlflow.db', dst_db)
    
    # Update paths in sqlite
    conn = sqlite3.connect(dst_db)
    conn.execute("UPDATE experiments SET artifact_location = replace(artifact_location, 'file:G:/Mi unidad/MAIA/Proyecto_Desarrollo_de_Soluciones/MicroP1MAIA/mlruns', '/data/mlruns')")
    conn.execute("UPDATE runs SET artifact_uri = replace(artifact_uri, 'file:G:/Mi unidad/MAIA/Proyecto_Desarrollo_de_Soluciones/MicroP1MAIA/mlruns', '/data/mlruns')")
    conn.commit()
    
    print("Experiments:", conn.execute("SELECT experiment_id, name, artifact_location FROM experiments").fetchall())
    conn.close()
    
    # Copy mlruns
    dst_mlruns = os.path.join(export_dir, 'mlruns')
    if os.path.exists(dst_mlruns):
        shutil.rmtree(dst_mlruns)
    shutil.copytree('mlruns', dst_mlruns)
    
    # Tar it
    tar_path = os.path.join(os.getcwd(), 'mlflow_data.tar.gz')
    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(dst_db, arcname='mlflow.db')
        tar.add(dst_mlruns, arcname='mlruns')
    
    size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"Created {tar_path} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    prepare_mlflow()
