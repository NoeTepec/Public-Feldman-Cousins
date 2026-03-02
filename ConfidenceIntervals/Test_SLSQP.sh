#!/bin/bash

# Retardo en base al numero de job
#BASE_DELAY=$(( ($3 % 50)*5 ))

# Retardo aleatorio pequeño (0-5 segundos)
#RANDOM_DELAY=$((RANDOM % 15))

# Retardo total
#TOTAL_DELAY=$((BASE_DELAY + RANDOM_DELAY))

# Retardo  Total
#sleep $TOTAL_DELAY

mkdir -p /tmp/envs/
mkdir -p /tmp/Tools/
mkdir -p /tmp/Test_results/ExtendedKFC/

# Copia del ambiente en un archivo .tar.gz y el souce
cp /eos/user/n/ntepecti/miniconda3/envs/py37.tar /tmp/envs/
cp /eos/user/n/ntepecti/miniconda3/etc/profile.d/conda.sh /tmp/envs/

# Copia de los scripts auxiliares y script principal
cp /eos/user/n/ntepecti/PhD/Feldman_Cousins/Tools/tools.py /tmp/Tools/
cp /eos/user/n/ntepecti/PhD/Feldman_Cousins/Tools/common_tools.py /tmp/Tools/
cp /eos/user/n/ntepecti/PhD/Feldman_Cousins/Tools/customStats.py /tmp/Tools/
cp /eos/user/n/ntepecti/PhD/Feldman_Cousins/Tools/SLSQP_zfit.py /tmp/Tools/
cp /eos/user/n/ntepecti/PhD/Feldman_Cousins/Tools/histos_weighted.py /tmp/Tools/
cp /eos/user/n/ntepecti/PhD/Feldman_Cousins/Tools/plot_tools.py /tmp/Tools/
cp /eos/user/n/ntepecti/PhD/Feldman_Cousins/Tools/ks_test.py /tmp/Tools/
cp /eos/user/n/ntepecti/PhD/Feldman_Cousins/Tools/SLSQP_FC2_try_parser.py /tmp/Tools/

#Descomprimimos el ambiente
cd /tmp/envs/
tar -xvzf py37.tar
cd /tmp/

# Cargamos conda
#source /eos/user/j/josuerau/miniconda3/etc/profile.d/conda.sh
source /tmp/envs/conda.sh
# Activamos el entorno en /eos
conda activate /tmp/envs/py37

# Ejecutamos el script
python /tmp/Tools/SLSQP_FC2_try_parser.py --mu $1 --sigma $2 --CL $3 --N_MC $4 --it_label $5 

