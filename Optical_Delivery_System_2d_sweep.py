import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import differential_evolution
import sympy as sym
import csv as csv
from mpl_toolkits.mplot3d import axes3d, Axes3D
from copy import copy

def q(R,Y,n,w):
    inverse_q = complex(1/R,-Y/(np.pi*n*w**2))
    return 1/inverse_q

def prop_mat(d):
    abcd = np.matrix([[1,d],[0,1]])
    return abcd
    

def lens_mat(f):
    abcd = np.matrix([[1,0],[-1/f,1]])
    return abcd

def full_optical_system(d_l,d_m,f_l,f_m):
    abcd_d_l = prop_mat(d_l)
    abcd_f_l = lens_mat(f_l)
    abcd_d_m = prop_mat(d_m)
    abcd_f_m = lens_mat(f_m)
    return np.dot(abcd_f_m,np.dot(abcd_d_m,np.dot(abcd_f_l,abcd_d_l)))

def beam_size_at_l(d_l):
    abcd = np.matrix([[1,d_l],[0,1]])
    x = np.dot(abcd,np.matrix([[q0],[1]]))
    q = x.item(0,0)/x.item(1,0)
    inv_q = 1.0/q
    w = np.sqrt(-Y/(np.imag(inv_q)*np.pi*n))
    return w



def col_const(d):
    d_l = d[0]
    d_m = d[1]
    return abs(np.real(1.0/((lambda x : x.item(0,0)/x.item(1,0))(np.dot(np.dot(np.dot(np.dot(np.matrix([[1,0],[-1/f_m,1]]),np.matrix([[1,d_m],[0,1]])),np.matrix([[1,0],[-1/f_l,1]])),np.matrix([[1,d_l],[0,1]])),np.matrix([[q0],[1]]))))))


def size_const(d):
    d_l = d[0]
    d_m = d[1]
    return abs(np.imag(1.0/((lambda x : x.item(0,0)/x.item(1,0))(np.dot(np.dot(np.dot(np.dot(np.matrix([[1,0],[-1/f_m,1]]),np.matrix([[1,d_m],[0,1]])),np.matrix([[1,0],[-1/f_l,1]])),np.matrix([[1,d_l],[0,1]])),np.matrix([[q0],[1]])))))+780e-9/(np.pi*w**2))

def const(d):
    return col_const(d)+size_const(d)

def focal_ratio_plane(f_l,f_d,factor):
    return -f_l/(2*factor)

if __name__== "__main__":
    w = 10e-3
    q0=q(100.0,780e-9,1.0,3e-6)
    n =1
    Y = 780e-9
    read = False
    num_of_points = 10
    f_l = -15e-3
    f_m = 25.1e-3
    w = 10e-3
    q0=q(100.0,780e-9,1.0,3e-6)
    f_l_lim = (-80e-3,-1e-3)
    f_m_lim = (1e-3,50e-3)
    f_l_array =np.linspace(f_l_lim[0],f_l_lim[1],num_of_points)
    f_m_array = np.linspace(f_m_lim[0],f_m_lim[1],num_of_points)
    counter = 0
    error_list = []

    total_img = []
    d_l_img = []
    d_m_img = []
    beam_size_img = []
    counter = 0
    if not read:

        for i in f_m_array:
            f_m =i
            total_row = []
            d_l_row = []
            d_m_row = []
            beam_size_row = []
            for j in f_l_array:
                f_l = j
                bounds = [(0,-2*f_l),(0,2*f_m)]
                success = False
                while not success:
                    ret = differential_evolution(const, bounds, tol =1e-6, maxiter = 10000)
                    success = ret.success
                    if success ==False:
                        error_list.append(ret)
                    if ret.fun >1e-9:
                        ret.x = [0,0]
                print("no "+str(counter))
                print(ret)
                print(ret.x[0]+ret.x[1])
                counter+=1

                total_row.append(ret.x[0]+ret.x[1])
                d_l_row.append(ret.x[0])
                d_m_row.append(ret.x[1])
                beam_size_row.append(beam_size_at_l(ret.x[0]))


            total_img.append(total_row)
            d_l_img.append(d_l_row)
            d_m_img.append(d_m_row)
            beam_size_img.append(beam_size_row)

        total_img = np.array(total_img)
        d_l_img = np.array(d_l_img)
        d_m_img = np.array(d_m_img)
        beam_size_img = np.array(beam_size_img)
            

        for img,name in zip([total_img,d_l_img,d_m_img,beam_size_img],["Total","DL","DM","Beam"]):
            with open('simulation_results50_2_debug'+name+'.csv', mode='w', newline = '') as simulation_file:
                pixel_writer = csv.writer(simulation_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in img:
                    pixel_writer.writerow(row)

    else:
        imgs = {}
        for name in ["Total","DL","DM","Beam"]:
            img = []
            with open(r'simulation_results50_small'+name+'.csv', mode='r') as simulation_file:
                pixel_reader = csv.reader(simulation_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                x = list(pixel_reader)
                img = np.array(x).astype("float")
                imgs[name]=img
        total_img =imgs["Total"]
        d_l_img =imgs["DL"]
        d_m_img =imgs["DM"]
        beam_size_img = imgs["Beam"]



     


    
    sf = 1000
    total_img*=sf
    d_l_img*=sf
    d_m_img*=sf
    beam_size_img*=sf

    f_l_array = [i*sf for i in f_l_array]
    f_m_array = [i*sf for i in f_m_array]
    X,Y = np.meshgrid(f_l_array, f_m_array)
    Z_1 = focal_ratio_plane(X,Y,1)
    Z_2 = focal_ratio_plane(X,Y,2)
    Z_4 = focal_ratio_plane(X,Y,4)
    beam_size_img_1 = copy(beam_size_img)
    beam_size_img_2 = copy(beam_size_img)
    beam_size_img_4 = copy(beam_size_img)
    total_img_1 = copy(total_img)
    total_img_2 = copy(total_img)
    total_img_4 = copy(total_img)
    col,row = np.shape(beam_size_img)
    for i in range(col):
        for j in range(row):
            if beam_size_img[i,j]>Z_1[i,j]:
                beam_size_img_1[i,j] = 0
                total_img_1[i,j]=0
            if beam_size_img[i,j]>Z_2[i,j]:
                beam_size_img_2[i,j] = 0
                total_img_2[i,j]=0
            if beam_size_img[i,j]>Z_4[i,j]:
                beam_size_img_4[i,j] = 0
                total_img_4[i,j]=0

    Z_1[Z_1>10] =np.nan
    Z_2[Z_2>10] =np.nan
    Z_4[Z_4>10] =np.nan
    # fig = plt.figure()
    # ax = Axes3D(fig)
    params = {'mathtext.default': 'regular' }     
    plt.rcParams.update(params)
    fig = plt.figure()

    beam_size_line_legend = lines.Line2D([0],[0], linestyle="none", c='r', marker = 'o')
    d2_line_legend = lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'$f_{l}$ (mm)')
    ax.set_ylabel(r'$f_{m}$ (mm)')
    ax.set_zlabel('size (mm)')
    ax.plot_surface(X, Y,beam_size_img,color = 'red', alpha = .5, linewidth =0,zorder =1 )
    ax.legend([beam_size_line_legend],[r'Beam Width w at Lens'])
    ax.view_init(elev=30., azim=30)

    fig_cut_1 = plt.figure()
    ax_cut_1 = fig_cut_1.add_subplot(111, projection='3d')
    ax_cut_1.set_xlabel(r'$f_{l}$ (mm)')
    ax_cut_1.set_ylabel(r'$f_{m}$ (mm)')
    ax_cut_1.set_zlabel('size (mm)')
    ax_cut_1.plot_surface(X, Y,beam_size_img_1,color = 'red', alpha = .5, linewidth =0,zorder =1 )
    ax_cut_1.legend([beam_size_line_legend,d2_line_legend], [r'Beam Width w at Lens [w<$D_{l}$/2]',r'$D_{l}$/2 (for $D_{l}$ =$f_{l}$)'], numpoints = 1)
    ax_cut_1.plot_surface(X,Y,Z_1,alpha = .4 , color = 'blue', zorder = 1)
    ax_cut_1.view_init(elev=30., azim=30)


    # fig_cut_2 = plt.figure()
    # ax_cut_2 = fig_cut_2.add_subplot(111, projection='3d')
    # ax_cut_2.set_xlabel(r'$f_{l}$ (mm)')
    # ax_cut_2.set_ylabel(r'$f_{m}$ (mm)')
    # ax_cut_2.set_zlabel('size (mm)')
    # ax_cut_2.plot_surface(X, Y,beam_size_img_2,color = 'red', alpha = .5, linewidth =0,zorder =1 )
    # ax_cut_2.legend([beam_size_line_legend,d2_line_legend], [r'Beam Width w at Lens [w<$D_{l}$/2]',r'$D_{l}$/2 (for $D_{l}$ =$f_{l}/2$)'], numpoints = 1)
    # ax_cut_2.plot_surface(X,Y,Z_2,alpha = .4 , color = 'blue', zorder = 1)
    # ax_cut_2.view_init(elev=30., azim=30)

    # fig_cut_4 = plt.figure()
    # ax_cut_4 = fig_cut_4.add_subplot(111, projection='3d')
    # ax_cut_4.set_xlabel(r'$f_{l}$ (mm)')
    # ax_cut_4.set_ylabel(r'$f_{m}$ (mm)')
    # ax_cut_4.set_zlabel('size (mm)')
    # ax_cut_4.plot_surface(X, Y,beam_size_img_4,color = 'red', alpha = .5, linewidth =0,zorder =1 )
    # ax_cut_4.legend([beam_size_line_legend,d2_line_legend], [r'Beam Width w at Lens [w<$D_{l}$/2]',r'$D_{l}$/2 (for $D_{l}$ =$f_{l}/4$)'], numpoints = 1)
    # ax_cut_4.plot_surface(X,Y,Z_2,alpha = .4 , color = 'blue', zorder = 1)
    # ax_cut_4.view_init(elev=30., azim=30)


    # fig_total = plt.figure()
    # ax_total = fig_total.add_subplot(111, projection='3d')
    # ax_total.set_xlabel(r'$f_{l}$ (mm)')
    # ax_total.set_ylabel(r'$f_{m}$ (mm)')
    # ax_total.set_zlabel('size (mm)')
    # ax_total.plot_surface(X, Y,total_img,color = 'red', alpha = .5, linewidth =0,zorder =1 )
    # ax_total.legend([beam_size_line_legend], [r'Total Distance'], numpoints = 1)
    # ax_total.view_init(elev=30., azim=30)

    # fig_total_1 = plt.figure()
    # ax_total_1 = fig_total_1.add_subplot(111, projection='3d')
    # ax_total_1.set_xlabel(r'$f_{l}$ (mm)')
    # ax_total_1.set_ylabel(r'$f_{m}$ (mm)')
    # ax_total_1.set_zlabel('size (mm)')
    # ax_total_1.plot_surface(X, Y,total_img_1,color = 'red', alpha = .5, linewidth =0,zorder =1 )
    # ax_total_1.legend([beam_size_line_legend], [r'Total Distance'], numpoints = 1)
    # ax_total_1.view_init(elev=30., azim=30)

    # fig_total_2 = plt.figure()
    # ax_total_2 = fig_total_2.add_subplot(111, projection='3d')
    # ax_total_2.set_xlabel(r'$f_{l}$ (mm)')
    # ax_total_2.set_ylabel(r'$f_{m}$ (mm)')
    # ax_total_2.set_zlabel('size (mm)')
    # ax_total_2.plot_surface(X, Y,total_img_2,color = 'red', alpha = .5, linewidth =0,zorder =1 )
    # ax_total_2.legend([beam_size_line_legend], [r'Total Distance'], numpoints = 1)
    # ax_total_2.view_init(elev=30., azim=30)

    # fig_total_4 = plt.figure()
    # ax_total_4 = fig_total_4.add_subplot(111, projection='3d')
    # ax_total_4.set_xlabel(r'$f_{l}$ (mm)')
    # ax_total_4.set_ylabel(r'$f_{m}$ (mm)')
    # ax_total_4.set_zlabel('size (mm)')
    # ax_total_4.plot_surface(X, Y,total_img_4,color = 'red', alpha = .5, linewidth =0,zorder =1 )
    # ax_total_4.legend([beam_size_line_legend], [r'Total Distance'], numpoints = 1)
    # ax_total_4.view_init(elev=30., azim=30)

    # fig_total_debug = plt.figure()
    # ax_total_debug = fig_total_debug.add_subplot(111)
    # im_total_debug = ax_total_debug.imshow(total_img_4, extent=[sf*f_l_lim[0], sf*f_l_lim[1],sf*f_m_lim[0],sf*f_m_lim[1]], origin = 'lower')
    # ax_total.set_title("Total Distance (mm)")
    # ax_total.set_xlabel("fl (mm)")
    # ax_total.set_ylabel("fm (mm)")
    # fig_total.colorbar(im_total_debug, ax=ax_total_debug)

    # cset = ax.contour(X, Y, beam_size_img, 200, extend3d=True)
    # cset 
    # ax.clabel(cset, fontsize=9, inline=1)

    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')

    plt.show()


