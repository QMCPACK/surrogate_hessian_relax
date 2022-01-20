#!/usr/bin/sh

psdir='./pseudos'
mkdir -p $psdir
cd $psdir

# download ccECP pseudopotentials
wget https://pseudopotentiallibrary.org/recipes/H/ccECP/H.ccECP.xml
wget https://pseudopotentiallibrary.org/recipes/H/ccECP/H.upf
wget https://pseudopotentiallibrary.org/recipes/C/ccECP/C.ccECP.xml
wget https://pseudopotentiallibrary.org/recipes/C/ccECP/C.upf

# download ultrasoft GBRV pseudopotentials
wget http://www.physics.rutgers.edu/gbrv/all_pbe_UPF_v1.5.tar.gz
tar -xvvf all_pbe_UPF_v1.5.tar.gz
rename "_" "." # rename for nexus
rm all_pbe_UPF_v1.5.tar.gz
