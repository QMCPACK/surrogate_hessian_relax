#!/usr/bin/env python3


from numpy import array, sin, cos, pi, exp, diag, mean

from surrogate_classes import mean_distances, bond_angle, distance

harmonic_a = lambda p,a: p[1]*(a-p[0])**2
# from Nexus
morse = lambda p,r: p[2]*((1-exp(-(r-p[0])/p[1]))**2-1)+p[3]

# test H2 molecule
pos_H2 = array('''
0.00000        0.00000        0.7
0.00000        0.00000       -0.7
'''.split(),dtype=float).reshape(-1,3)
elem_H2 = 'H H'.split()
def forward_H2(pos):
    r = distance(pos[0], pos[1])
    return [r]
#end def
def backward_H2(params):
    H1 = params[0]*array([0.0, 0.0, 0.5])
    H2 = params[0]*array([0.0, 0.0,-0.5])
    return array([H1, H2])
#end def
hessian_H2 = array([[1.0]])
def pes_H2(params):  # inaccurate; for testing
    r, = tuple(params)
    V = 0.0
    V += morse([1.4, 1.17, 0.5, 0.0], r)
    return V
#end def
def alt_pes_H2(params):  # inaccurate; for testing
    r, = tuple(params)
    V = 0.0
    V += morse([1.35, 1.17, 0.6, 0.0], r)
    return V
#end def
def get_structure_H2():
    from surrogate_classes import ParameterStructure
    return ParameterStructure(forward = forward_H2, backward = backward_H2, pos = pos_H2, elem = elem_H2)
#end def
def get_hessian_H2():
    from surrogate_classes import ParameterHessian
    return ParameterHessian(hessian = hessian_H2)
#end def


# test H2O molecule
pos_H2O = array('''
0.00000        0.00000        0.11779
0.00000        0.75545       -0.47116
0.00000       -0.75545       -0.47116
'''.split(),dtype=float).reshape(-1,3)
elem_H2O = 'O H H'.split()
def forward_H2O(pos):
    r_OH = mean_distances([(pos[0], pos[1]), (pos[0], pos[2])])
    a_HOH = bond_angle(pos[1], pos[0], pos[2])
    return [r_OH, a_HOH]
#end def
def backward_H2O(params):
    r_OH = params[0]
    a_HOH = params[1]*pi/180
    O = [0., 0., 0.]
    H1 = params[0]*array([0.0, cos((pi-a_HOH)/2), sin((pi-a_HOH)/2)])
    H2 = params[0]*array([0.0,-cos((pi-a_HOH)/2), sin((pi-a_HOH)/2)])
    return array([O, H1, H2])
#end def
hessian_H2O = array([[1.0, 0.2],
                     [0.2, 0.5]])  # random guess for testing purposes
hessian_real_H2O = array('''
1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 2.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0
0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 4.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0
0.0 2.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 3.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0
0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.2
'''.split(), dtype = float).reshape(9, 9)
def pes_H2O(params):
    r, a = tuple(params)
    V = 0.0
    V += morse([0.95789707, 0.5, 0.5, 0.0], r)
    V += harmonic_a([104.119, 0.5], a)
    return V
#end def
def get_structure_H2O():
    from surrogate_classes import ParameterStructure
    return ParameterStructure(forward = forward_H2O, backward = backward_H2O, pos = pos_H2O, elem = elem_H2O)
#end def
def get_hessian_H2O():
    from surrogate_classes import ParameterHessian
    return ParameterHessian(hessian = hessian_H2O)
#end def
def job_H2O_pes(structure, path, sigma, **kwargs):
    p = structure.params
    value = pes_H2O(p)
    return [(path, value, sigma)]
#end def
def analyze_H2O_pes(path, job_data = None, **kwargs):
    for row in job_data:
        if path == row[0]:
            return row[1], row[2]
        #end if
    #end for
    return None
#end def
def get_surrogate_H2O():
    from surrogate_classes import TargetParallelLineSearch
    srg = TargetParallelLineSearch(
        structure = get_structure_H2O(),
        hessian = get_hessian_H2O(),
        M = 25,
        window_frac = 0.5)
    params0 = srg.get_shifted_params(0)
    params1 = srg.get_shifted_params(1)
    values0 = [pes_H2O(p) for p in params0]
    values1 = [pes_H2O(p) for p in params1]
    srg.load_results(values = [values0, values1], set_target = True)
    return srg
#end def

# test GeSe monolayer
#                    a     b     x       z1       z2
params_GeSe = array([4.26, 3.95, 0.4140, 0.55600, 0.56000])
elem_GeSe = 'Ge Ge Se Se'.split()
def forward_GeSe(pos, axes):
    Ge1, Ge2, Se1, Se2 = tuple(pos)
    a = axes[0,0]
    b = axes[1,1]
    x = mean([Ge1[0], Ge2[0] - 0.5])
    z1 = mean([Ge1[2], 1 - Ge2[2]])
    z2 = mean([Se2[2], 1 - Se1[2]])
    return [a, b, x, z1, z2]
#end def
def backward_GeSe(params):
    a, b, x, z1, z2 = tuple(params)
    Ge1 = [x,       0.25, z1]
    Ge2 = [x + 0.5, 0.75, 1 - z1]
    Se1 = [0.5,     0.25, 1 - z2]
    Se2 = [0.0,     0.75, z2]
    axes = diag([a, b, 20.0])
    pos = array([Ge1, Ge2, Se1, Se2])
    return pos, axes
#end def
# random guess for testing purposes
hessian_GeSe = array([[1.0,  0.5,  40.0,  50.0,  60.0],
                      [0.5,  2.5,  20.0,  30.0,  10.0],
                      [40.0, 20.0, 70.0,  30.0,  10.0],
                      [50.0, 30.0, 30.0, 130.0,  90.0],
                      [60.0, 10.0, 10.0,  90.0, 210.0]])



# resampled normally distributed data
Gs_N200_M7 = array('''
0.703364 -1.186137 -1.763959 -1.169658 1.815397 0.794752 0.824720 
-0.251263 -0.352322 -1.126102 -0.604460 -0.844856 0.501666 0.493686 
-1.205878 1.435354 -1.533030 1.158740 1.034258 -0.788764 -0.055269 
0.070773 0.175632 -1.328132 -0.812551 0.572434 0.042435 0.426912 
-1.110780 -0.249747 -0.575956 0.114005 -2.152316 -0.228293 0.085509 
-1.330517 1.341788 -0.192607 0.431155 -1.910323 0.154812 1.694543 
1.301801 2.316676 -0.144464 0.466756 0.568335 1.202510 0.014773 
-1.465544 0.502388 0.527962 1.027726 0.782742 0.400225 0.538037 
0.217018 -1.713586 0.347273 -0.911403 0.162836 0.854430 0.065537 
0.229157 0.418065 -0.437793 -0.487860 -0.935640 -1.299435 -0.294986 
0.588626 0.449454 -1.196819 0.134440 0.500840 -0.358673 0.766368 
-0.931242 0.628175 1.578036 1.969466 -0.795563 0.472256 0.477718 
-0.143606 0.244128 -1.013039 0.689445 -1.066076 -1.725752 0.078912 
1.507204 -2.074634 0.322966 -0.279504 0.588732 -0.483143 1.227501 
-0.159345 1.066015 -0.543481 -0.273738 -0.018459 0.252087 0.091283 
-0.953924 -0.134242 -0.520108 0.012531 -0.887133 0.618472 -1.267098 
-0.579064 1.197260 -1.058323 1.424631 1.007222 0.358788 -0.039058 
0.074029 -0.977219 0.445223 0.823262 0.017584 -0.049317 -0.379005 
-0.780307 -2.396043 -0.730487 -0.477390 -0.552751 1.215461 -0.402970 
0.954802 0.098818 0.020284 1.000029 -0.543290 1.556811 -2.996884 
0.717299 -0.836932 -0.132646 0.381092 -1.060019 0.122711 1.451803 
-0.640169 0.081480 -0.425924 0.228384 -1.792736 -0.718514 0.557250 
0.897201 1.108213 0.632906 -0.777191 -0.054309 0.363085 -0.248579 
1.085563 -1.257829 0.310274 -0.639929 0.249955 -1.382891 -0.843118 
1.018726 0.372610 -0.057655 -1.229712 -0.408168 0.100365 0.824778 
-1.767408 -1.488234 0.371417 -0.301248 0.037813 -0.543092 -0.550050 
1.040229 -1.156375 2.076503 -0.508958 -0.684074 -0.368156 0.255705 
-0.945274 -0.137725 -0.501251 1.848483 -0.555349 0.395152 -1.424211 
0.489956 0.340145 -1.143131 -0.267478 0.164491 1.614948 0.607109 
-0.796952 1.939116 0.834003 -0.156337 0.028238 -1.658432 1.885640 
0.692051 2.411910 0.663040 -0.038033 1.234208 1.194370 -0.271674 
-0.725860 0.058264 -0.870238 -0.237182 -1.495710 -2.100849 0.197393 
0.159780 -0.319385 -0.291632 0.196433 -0.774237 -1.043405 -1.414815 
-0.890207 0.784695 -0.438945 -0.283920 1.811329 -0.133351 0.448276 
0.251463 -1.299643 0.349288 1.335740 0.336007 0.166185 1.953064 
0.176220 0.328413 -1.076256 -1.264913 1.322628 -0.985915 0.727727 
0.734984 -0.054962 -0.348743 1.596476 1.473778 1.333215 -1.202976 
-0.787011 -0.513615 0.850655 -0.418900 -0.179525 0.126006 -0.281515 
-1.508399 -0.226939 1.247584 -2.215555 -1.086675 1.231869 -1.569804 
0.396400 1.619960 -0.436116 -0.283097 -1.798144 1.007003 2.323567 
-0.012347 -0.235260 -0.465460 -1.551464 0.111473 -0.499988 0.430274 
-0.576026 1.015583 1.180262 0.073210 -2.652759 0.602338 -0.602022 
0.315806 -1.218275 -0.319277 -0.075865 -0.330046 1.471219 0.869526 
-1.553109 0.523736 -0.813140 0.062242 -0.105304 -0.460764 1.012730 
0.521964 -1.487004 -1.162915 0.824431 0.950007 -2.321529 -0.555982 
0.482038 -1.228050 -1.053045 -0.039151 0.630817 -0.433563 -1.391248 
-0.357328 0.589713 0.327144 1.674743 1.267933 -1.127315 -0.200406 
-0.499801 1.915533 -0.744806 0.864902 0.940895 -0.991135 0.741969 
-0.720215 2.029828 -1.535780 -0.399506 -1.650718 0.368974 -1.092392 
-1.236857 1.203296 -0.372978 -1.196185 1.506771 -0.019686 -1.348424 
0.792694 -0.269185 -0.099109 -0.215290 -0.766336 0.522152 1.459370 
1.616206 -0.546700 1.232107 1.368972 0.228799 -0.586168 0.657967 
0.390038 -0.590828 -2.386434 -0.442417 -1.150673 0.611584 -0.010623 
1.518331 -0.069890 1.835902 -0.112260 -0.251004 -0.432407 0.524892 
0.446823 1.793103 1.197498 0.732892 0.037010 0.601903 0.557700 
-1.579173 1.300507 -0.738038 1.232515 1.712402 -0.404404 -0.607473 
0.450945 0.781938 0.468530 -0.247119 0.101052 -0.389055 -1.034914 
0.112536 0.099637 -1.301783 -0.039429 -0.037169 0.275224 0.782089 
-0.973241 0.498418 0.240069 -0.676455 -0.395279 -0.698010 0.782244 
-1.628349 -0.471704 1.305955 0.069451 0.127069 0.809073 -0.262988 
-1.352523 0.792904 0.007351 1.185062 0.483908 0.839234 0.479352 
0.843189 0.003400 -0.327981 1.848854 0.043876 -0.602856 0.484536 
1.493175 -1.827523 -0.587222 -0.778402 1.127385 -0.297450 -1.293115 
0.135243 -1.971791 1.198190 1.283809 0.817124 -1.917487 -0.479708 
0.862235 1.065572 0.736476 -0.124931 -0.213850 -0.246377 -1.493588 
1.103652 -0.735816 1.939359 -0.102345 0.386097 -1.188564 -0.133847 
0.478442 1.174194 0.417347 -0.898711 -0.097774 0.310439 -1.156731 
-0.995089 -1.106503 0.005497 -0.088000 0.006124 0.166960 0.118199 
0.626601 0.683360 -0.432558 0.290182 -0.156704 0.131995 -1.608062 
-1.060304 -0.277138 0.823433 -0.069390 -0.451049 -0.938098 -0.656411 
-2.448282 -0.727952 1.785295 -0.604069 2.574547 -0.724815 -0.806251 
-0.103382 0.983530 -1.766896 0.242764 -1.019678 0.405259 -1.489066 
-0.384312 0.181570 -0.577817 -0.022625 0.644557 -0.358164 0.373690 
0.062787 0.317604 1.445442 0.137709 -0.468568 -0.967278 0.696949 
0.808686 1.507436 -0.083858 -0.453345 -1.061160 -0.010072 1.142089 
-1.320856 -2.164973 1.489600 1.147017 0.808122 -0.615067 0.091589 
-0.523773 0.522252 0.773262 0.302695 1.116450 1.893534 -0.509278 
1.138474 0.662635 0.607651 0.775873 -0.256715 -1.198288 0.576999 
-1.123817 2.216662 0.807298 -0.186321 1.842465 0.694473 -0.855191 
-0.312036 -0.320361 -0.702030 0.562965 1.357866 0.544611 -1.168216 
0.291175 1.525420 -0.781630 1.009957 -0.274833 -0.129864 1.593744 
-1.404063 -0.230874 0.987916 -0.191553 1.245202 -0.200049 -0.116515 
-0.999233 1.924690 -1.284108 1.678151 -0.290467 -0.015245 -0.296421 
-0.491104 0.991214 1.782462 -0.438015 -0.437691 0.261227 0.323717 
1.676331 0.931640 1.259691 0.433143 0.387783 0.224046 -1.395712 
-0.703937 -1.015293 0.680128 -0.395039 1.779956 0.644663 -0.841585 
-0.550442 1.342052 -0.550427 0.554967 -1.050647 -1.738159 0.007911 
0.258605 0.234402 0.300518 -0.772297 -0.595169 0.757208 -1.406795 
0.617094 1.045694 -1.290859 -0.938630 0.855891 -0.163793 0.043147 
0.343124 -0.450949 2.268464 0.890277 0.158599 0.433975 -1.159910 
1.193372 -2.174435 -1.229208 1.543071 -1.139786 -1.301680 -0.912963 
0.209710 1.389922 0.359041 0.063694 -0.085798 -0.091022 -1.485686 
-1.612105 0.816145 0.524234 0.505148 0.823380 0.188416 -1.722275 
-0.750154 1.405684 0.822539 0.536679 0.352180 0.009156 -1.379445 
1.589241 1.842342 0.965406 1.003678 -2.176170 0.992046 -1.776101 
0.414365 -0.954006 -0.710746 1.233066 -0.268482 -1.381946 -0.906853 
-1.867055 -1.021391 -0.863102 0.506538 0.436673 1.186974 2.760487 
0.207598 -0.567531 -0.706615 0.791588 -0.002134 -1.030221 1.273854 
0.129919 2.467808 -0.562256 0.236774 -1.338554 -0.917171 -0.938751 
-0.150236 0.218032 0.481388 -0.557174 1.305371 0.018283 -0.904232 
0.816508 0.178874 -0.406176 -0.721822 -1.157864 0.683962 0.900194 
-0.998549 -1.645228 0.157738 0.648298 0.710846 1.478251 -1.927036 
0.919081 -0.296954 0.155391 -0.934386 1.923796 1.992032 -0.224161 
1.109876 -0.235195 1.342097 0.652507 -1.286178 0.925754 -1.442364 
-0.354660 1.111155 -1.056762 0.573363 -0.526423 0.606616 -0.337065 
-0.247147 0.584602 -0.400189 -1.011218 1.313763 0.275262 0.718636 
-1.123434 -0.513053 0.501945 -0.679632 0.911030 -1.360480 -0.673193 
-0.417866 -0.528705 -1.474387 1.784902 -1.385613 1.856434 -2.068550 
0.383989 -1.004892 -0.729469 -0.323005 -0.159383 -0.788105 0.435939 
-2.354846 -0.222201 0.613990 -0.213366 -0.264652 1.759245 0.238009 
-0.457440 -0.320013 0.772403 0.740003 0.607383 -1.175896 0.363790 
-0.008288 0.450862 0.694701 0.830339 2.241631 -0.455330 0.045068 
-0.389082 1.152080 -0.765442 -0.556965 -0.084524 -2.191910 0.200578 
0.739202 -0.579065 1.478615 1.407373 -0.565333 -0.372382 -1.885613 
-1.109243 0.311912 -0.840493 1.077988 -0.489288 1.354002 0.870082 
0.432302 -0.368326 1.242332 -0.898720 0.402653 -0.854414 -1.208983 
-1.162958 -1.261596 0.684492 1.771045 -2.213724 -0.028262 -0.753717 
1.181105 -0.451088 -1.144053 -1.054180 -0.527759 0.183229 0.692925 
0.271783 -0.178489 -1.359130 0.610474 -0.941190 0.347221 -1.739609 
-0.270695 -0.694513 0.885122 0.143781 -1.274560 1.132093 1.552943 
-0.722649 -0.335310 -0.009100 -0.434350 1.294646 -1.370524 0.163792 
-2.243796 0.518299 0.288581 0.157036 -0.105726 0.300993 0.973381 
-1.161017 0.843680 0.427907 -0.859007 -0.881306 -1.681602 0.481659 
1.008895 1.227666 -1.522128 1.169658 0.209103 0.311544 -0.573884 
-0.448628 -2.525361 -0.481624 0.073688 -1.328310 1.122452 0.405371 
0.791226 2.364479 -0.224028 0.548086 -2.323403 -0.513721 0.763940 
0.145947 -0.108968 -0.291802 -0.875027 -1.071140 -0.667331 0.354060 
1.000820 -0.543771 -1.135798 1.467657 0.326709 2.124726 1.802076 
0.541277 -0.330342 1.331189 0.277335 1.125227 -0.484740 1.346803 
1.689938 0.269054 -0.704119 -1.281026 -0.505312 -0.068153 1.201502 
-1.686588 0.970175 1.388735 -0.878344 0.494919 0.891528 -0.487951 
0.960434 -0.374082 -0.753649 1.559125 1.108408 -0.392025 -0.076283 
0.461971 1.167916 0.772131 -0.714258 -1.078419 0.421631 -0.372079 
-1.256068 0.754296 -0.000704 0.309246 0.145197 0.176814 0.927982 
3.932430 0.301709 -2.139134 0.681115 -1.641921 0.239599 -0.018069 
0.105447 -1.377600 0.021583 -1.154114 -0.361382 0.635633 -1.682405 
-0.680280 -0.848926 -0.795811 -1.139101 1.250585 0.258300 0.492934 
-0.517119 0.585097 -0.332246 0.767088 -0.417226 -0.515287 -0.457140 
0.431424 1.867177 -0.585676 -0.504178 -1.182859 -0.102238 -0.319216 
0.329200 1.597382 -0.602206 1.072467 0.018209 -0.883208 1.299254 
-0.757389 -0.409939 -0.244319 0.987938 1.864209 2.542839 0.930055 
0.294356 -0.487440 -2.134684 1.161526 0.587018 1.142690 -0.591943 
0.135479 -1.734625 -0.021450 -0.810960 -1.383159 -0.790265 0.006637 
-0.618494 0.915408 -0.839621 0.446954 -1.782658 -0.593248 0.182054 
0.933382 1.818280 0.673599 0.609236 0.410174 1.282024 -2.040189 
-1.121841 0.061186 0.439962 -0.901531 -1.284429 -0.506005 0.627485 
1.256385 0.321775 0.495017 -1.388514 0.497416 -1.538549 0.256710 
-0.367521 1.113420 0.068230 -0.284250 0.269500 -1.723635 -0.100961 
-3.519791 0.715167 0.154916 1.158336 -0.292414 -0.792169 0.544365 
1.328550 -0.174503 1.358190 -1.007658 0.856471 0.054038 -0.488718 
-0.291679 0.198696 -0.622601 1.912930 -2.892791 -0.767960 -2.462878 
0.421267 0.571676 -0.095771 2.483716 1.380152 -0.502559 -1.061435 
-0.508626 -2.220387 2.030256 -0.784298 0.722117 0.906199 -0.592164 
-0.552844 0.267199 0.525202 -1.487806 1.460600 0.325451 0.402462 
0.578843 -0.258205 0.038588 -1.598042 0.553691 -0.045340 -0.977464 
-0.236947 1.111945 1.494634 0.563144 0.589171 -0.942089 1.332926 
1.498214 1.292474 0.608482 -0.741920 2.571648 -0.077189 -0.282875 
1.748062 1.059780 -0.367631 -0.283898 0.029730 -0.365057 0.478507 
-0.437333 0.260710 -0.462527 -0.374864 2.968344 -0.289797 -1.502327 
1.510202 2.210206 0.153062 -2.129959 -0.274576 -0.086158 0.753442 
0.602019 0.389047 -0.574585 -0.239301 -0.647802 -0.189417 0.254682 
-0.172471 -1.014211 -1.934036 0.436446 -0.036482 0.916600 -0.356266 
-0.403194 -1.979303 -0.227945 0.135588 2.180943 -0.404430 0.296027 
-0.937888 0.216011 -0.405538 1.372900 -0.568571 0.322781 0.320513 
-0.986282 2.900643 0.328768 1.900032 0.757303 1.093198 -1.395413 
-0.461148 0.555199 0.061930 -1.846018 1.130982 0.827381 0.702648 
-0.700451 1.396331 -0.424938 -2.537416 -0.057175 -0.319024 -0.335389 
-0.183874 0.682927 0.039910 -0.309518 0.558009 -2.237180 -1.087873 
0.999739 0.534827 -0.783934 -1.552743 1.084964 -1.361997 1.067425 
1.573855 -0.117000 -0.064149 0.585813 1.197479 -2.233922 -1.022514 
0.902509 -0.513596 -1.440351 -0.506132 -1.740725 -1.474564 0.403109 
1.162804 -1.587994 -1.865740 0.150363 -1.164703 -0.227217 0.210398 
0.558862 -1.374320 -0.685145 0.092614 0.497920 0.594662 0.038293 
-0.023119 0.462564 -0.477067 -0.301233 -1.072171 -0.591412 0.874010 
-0.032893 -1.987064 0.643164 -0.042696 -0.386408 1.098967 0.003299 
-0.638926 -2.037183 -0.111740 2.098708 -0.798944 -0.571512 0.146815 
0.430765 -0.459851 1.310859 0.550442 0.425951 -0.627162 -0.855212 
1.013434 0.456319 0.412137 -0.557985 1.727521 -1.118597 1.608783 
0.725622 0.418119 1.039565 -0.327593 0.879424 -0.027633 -1.241763 
-0.076828 -1.044816 1.336309 -1.022163 -0.324097 1.169049 -0.365273 
0.456338 0.891435 -0.867075 1.453801 0.673608 0.675523 1.079618 
0.310334 -0.483523 -0.898151 -0.109777 -0.842999 0.276925 0.828162 
-2.608671 -0.963997 0.558172 -1.159346 -0.078836 -0.313740 0.599605 
-0.035851 0.650410 -0.356044 -2.763976 -0.422213 -0.629499 -0.786838 
0.517178 -1.152028 0.371989 2.308800 -0.281616 0.114252 0.571048 
0.000401 -1.329638 0.875179 -0.547619 0.185870 -0.171389 -1.007050 
-2.308857 0.867107 -0.242376 -1.768786 2.029829 -1.196773 0.239549 
-0.009313 -0.697478 -2.292251 -0.358822 -0.527450 0.258493 -1.100020 
2.204771 0.103619 1.442388 -0.626435 -0.888744 1.371220 1.101796 
0.135275 0.022144 -0.193058 0.994110 0.483387 0.564190 1.419475 
0.099075 1.714522 0.066849 -0.351209 -0.724503 -0.885567 0.423341 
-3.161330 1.679143 -1.200966 0.887853 -0.545337 -1.502706 -0.748493 
-0.889954 0.789635 0.882387 -0.938492 -0.606206 -1.802763 2.181509 
-0.031089 1.351732 1.267510 0.174596 -2.854278 1.843243 1.072557 
0.198824 -0.602352 -1.756950 0.370543 -0.371345 2.162031 -1.692823 
-0.206354 -0.191935 1.299092 0.395017 1.162509 0.553169 0.497992 
-0.904133 0.727203 -0.333388 -0.231292 1.042380 1.111640 1.249051 
-0.406082 0.644093 0.117351 -1.401056 -0.160014 -0.210272 -0.739391 
-0.689706 0.006336 0.666765 0.319331 -1.515724 3.801438 0.183464 
-0.517563 0.473361 -1.039603 1.995563 -1.971662 -0.692435 1.931064 
'''.split(), dtype = float).reshape(200, 7)
