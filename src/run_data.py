"""

Data container for loading a given run from database to feed to GUI, used for visualization

"""

# region Imports

import shutil
import numpy as np
from pathlib import Path
from src.db import get_run_samples, get_run_molecules
from src.intensity_matrix import IntensityMatrix as IM
from src.config_loader import ConfigLoader
from src.data_matrix import DataMatrix as DM
from src.db import connect
from src.utils import get_proj_db, get_run_dir
from src.db import insert_peak_batch, insert_im, insert_run, insert_molecule, insert_sample

# logging
import logging
logger = logging.getLogger(__name__)

# endregion

class RunData:
    def __init__(self, run_name: str, proj_name: str, cfg):

        self.proj_name = proj_name
        self.run_name = run_name
        self.run_type = None
        self.cfg = cfg
        self.failed_samples = None
        
        db_path = get_proj_db(proj_name)
        try:
            conn = connect(db_path)
            self.samples = {r['sample_name']: {k:v for k,v in dict(r).items() if k!='run_name'} 
                            for r in get_run_samples(conn, run_name)}
            self.molecules = {r["molecule_name"]: {k:v for k,v in dict(r).items() if k!='run_name'} 
                              for r in get_run_molecules(conn, run_name)}
        finally:
            conn.close()

        mols = []
        mzs = []
        rts = []
        for entry in self.molecules.values():
            mols.append(entry['molecule_name'])
            mzs.append(np.int64(entry['ion']))
            rts.append(entry['rt'])

        self.intensity_matrices = {}
        peaks = {}
        for sample_name in self.samples:
            matrix = IM.load_h5_object(sample_name, proj_name, run_name)
            peak_list = matrix.collect_data(mols, mzs, rts)
            for peak in peak_list:
                matrix.generate_spectra(peak,label=peak['molecule'],n_closest=10,save_spectra=True)
            peaks[sample_name] = peak_list
            self.intensity_matrices[sample_name] = matrix

        self.max_mz = np.nanmax(mz for im in self.intensity_matrices.values() for mz in im.unique_mzs if mz != 9999)

        self.data_matrix = DM(proj_name, run_name, peaks, self.samples, self.molecules, self.max_mz, self.cfg)

        # reassign detected molecules and get spectra for use
        sample_names = {v:k for k,v in self.data_matrix.sample_map.items()}
        mol_names = {v:k for k,v in self.data_matrix.mol_map.items()}
        for i in range(self.data_matrix.data['peak_idx'].shape[0]):         # rows/samples
            for j in range(self.data_matrix.data['peak_idx'].shape[1]):     # cols/molecules
                
                peak_idx = self.data_matrix.data['peak_idx'][i][j]
                if peak_idx == -1:
                    continue

                name = sample_names[i]
                molecule = mol_names[j]

                im = self.intensity_matrices[name]
                ion = np.int64(self.molecules[molecule]['ion'])

                peak_list = im.peak_dict[ion]
                peak = peak_list[peak_idx]

                peak['molecule'] = molecule

    # region Data loading/saving

    @classmethod
    def from_processing(cls, proj_name, run_name, samples, molecules, intensity_matrices, run_type, cfg):
        """
        Creates rundata object before anything has been saved to sql database 
        """
        obj = cls.__new__(cls)
        obj.proj_name = proj_name
        obj.run_name = run_name
        obj.samples = samples
        obj.molecules = molecules
        obj.intensity_matrices =intensity_matrices
        obj.run_type = run_type
        obj.cfg = cfg
        
        mols = []
        mzs = []
        rts = []
        for entry in obj.molecules.values():
            mols.append(entry['molecule_name'])
            mzs.append(np.int64(entry['ion']))
            rts.append(entry['rt'])

        input_dir = Path(cfg.get('input_dir'))
        input_type = cfg.get('input_type')

        obj.failed_samples = []
        peaks = {}
        for sample_name in obj.samples:
            matrix = obj.intensity_matrices.get(sample_name)
            if input_type == '.D':
                file_path = input_dir / 'mzML_files' / f"{sample_name}.mzML"
            else:
                file_path = input_dir / f"{sample_name}.mzML"
            if matrix is None:
                logger.warning(f"No IntensityMatrix found for {sample_name}, sample ws not processed successfully")
                obj.failed_samples.append([sample_name,file_path])
                continue
            peak_list = matrix.collect_data(mols, mzs, rts)
            for peak in peak_list:
                matrix.generate_spectra(peak,label=peak['molecule'],n_closest=10,save_spectra=True)
            peaks[sample_name] = peak_list

        obj.max_mz = max(mz for im in obj.intensity_matrices.values() for mz in im.unique_mzs if mz != 9999) + 1

        # create data matrix 
        obj.data_matrix = DM(proj_name, run_name, peaks, samples, molecules, obj.max_mz, obj.cfg)

        # reassign detected molecules
        sample_names = {v:k for k,v in obj.data_matrix.sample_map.items()}
        mol_names = {v:k for k,v in obj.data_matrix.mol_map.items()}
        for i in range(obj.data_matrix.data['peak_idx'].shape[0]):         # rows/samples
            for j in range(obj.data_matrix.data['peak_idx'].shape[1]):     # cols/molecules
                
                peak_idx = obj.data_matrix.data['peak_idx'][i][j]
                if peak_idx == -1:
                    continue

                name = sample_names[i]
                molecule = mol_names[j]

                im = obj.intensity_matrices[name]
                ion = np.int64(obj.molecules[molecule]['ion'])

                peak_list = im.peak_dict[ion]
                peak = peak_list[peak_idx]

                peak['molecule'] = molecule

        return obj

    def save_run(self, cfg):
        """
        saves run to sql/h5
        """
        
        conn = None
        run_dir = None

        try:
            # make run dir
            run_dir = get_run_dir(self.proj_name, self.run_name)
            run_dir.mkdir(parents=True,exist_ok=True)

            # save config
            cfg.save()

            # connect to db
            conn = connect(get_proj_db(self.proj_name))

            # inset ito db/h5
            with conn:
                # insert run to table
                insert_run(conn,self.run_name,self.run_type)

                # insert sample/molecule data to DB
                for row in self.samples.values():
                    insert_sample(conn,
                                    row['sample_name'],
                                    self.run_name,
                                    row['modelID'],
                                    row['group_name'],
                                    row['sex'],
                                    row['norm_factor'],
                                    row['injection_order'])

                for row in self.molecules.values():
                    insert_molecule(conn,
                                    row['molecule_name'],
                                    self.run_name,
                                    row['ion'],
                                    row['rt'],
                                    row['std'],
                                    row['casNo'])

                # generate intensity mtrices, save, collect data and save
                for im in self.intensity_matrices.values():
                    insert_im(conn,
                              im.sample_name,
                              self.run_name,
                              im.matrix_type,
                              im.noise_factor,
                              im.intensity_matrix.shape[0],
                              im.intensity_matrix.shape[1])
                    insert_peak_batch(conn, im, self.run_name)
                    im.save_h5_object(self.proj_name, self.run_name)

        except Exception:
            raise

        finally:
            if conn:
                conn.close()

    # endregion