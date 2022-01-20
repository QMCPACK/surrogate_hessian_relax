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
cp c_pbe_v1.2.uspp.F.UPF C.pbe_v1.2.uspp.F.upf
cp h_pbe_v1.4.uspp.F.UPF H.pbe_v1.4.uspp.F.upf
rm *.UPF
rm all_pbe_UPF_v1.5.tar.gz
